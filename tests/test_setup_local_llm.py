"""Pure-logic tests for setup_local_llm.py: sha256 verify/skip, config merge, argv building.

Network and zip extraction are stubbed everywhere -- no download ever touches the network
in these tests.
"""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import setup_local_llm as setup


class FakeResponse:
	"""Stand-in for the object urllib.request.urlopen returns."""

	def __init__(self, body: bytes, status: int = 200, headers=None):
		self._body = body
		self.status = status
		self.headers = headers if headers is not None else {"Content-Length": str(len(body))}

	def read(self, size: int = -1) -> bytes:
		if size is None or size < 0:
			chunk, self._body = self._body, b""
		else:
			chunk, self._body = self._body[:size], self._body[size:]
		return chunk

	def __enter__(self):
		return self

	def __exit__(self, *_exc):
		return False


class Sha256Tests(unittest.TestCase):
	def test_sha256_of_file_matches_hashlib(self):
		with tempfile.TemporaryDirectory() as tmp:
			path = Path(tmp) / "a.bin"
			data = b"hello world" * 100
			path.write_bytes(data)
			self.assertEqual(setup.sha256_of_file(path), hashlib.sha256(data).hexdigest())

	def test_verify_existing_true_only_on_hash_match(self):
		with tempfile.TemporaryDirectory() as tmp:
			path = Path(tmp) / "a.bin"
			path.write_bytes(b"content")
			good = hashlib.sha256(b"content").hexdigest()
			self.assertTrue(setup.verify_existing(path, good))
			self.assertFalse(setup.verify_existing(path, "0" * 64))
			self.assertFalse(setup.verify_existing(Path(tmp) / "missing.bin", good))


class DownloadFileTests(unittest.TestCase):
	def test_skips_download_when_destination_already_verifies(self):
		with tempfile.TemporaryDirectory() as tmp:
			dest = Path(tmp) / "file.zip"
			dest.write_bytes(b"already-here")
			expected = hashlib.sha256(b"already-here").hexdigest()
			opener = mock.Mock(side_effect=AssertionError("should not touch the network"))
			setup.download_file("http://example.test/file.zip", dest, expected, urlopen=opener)
			opener.assert_not_called()
			self.assertEqual(dest.read_bytes(), b"already-here")

	def test_downloads_to_part_then_renames_on_success(self):
		with tempfile.TemporaryDirectory() as tmp:
			dest = Path(tmp) / "file.bin"
			body = b"payload-bytes" * 10
			expected = hashlib.sha256(body).hexdigest()
			opener = mock.Mock(return_value=FakeResponse(body))
			progress = []
			setup.download_file("http://example.test/file.bin", dest, expected, urlopen=opener, on_progress=lambda *a: progress.append(a))
			self.assertTrue(dest.is_file())
			self.assertFalse(dest.with_name(dest.name + ".part").exists())
			self.assertEqual(dest.read_bytes(), body)
			self.assertTrue(progress)  # some progress was reported

	def test_sha_mismatch_deletes_part_and_raises_without_touching_dest(self):
		with tempfile.TemporaryDirectory() as tmp:
			dest = Path(tmp) / "file.bin"
			opener = mock.Mock(return_value=FakeResponse(b"wrong-bytes"))
			with self.assertRaisesRegex(RuntimeError, "sha256"):
				setup.download_file("http://example.test/file.bin", dest, "0" * 64, urlopen=opener)
			self.assertFalse(dest.exists())
			self.assertFalse(dest.with_name(dest.name + ".part").exists())

	def test_existing_part_is_resumed_with_a_range_request_on_206(self):
		with tempfile.TemporaryDirectory() as tmp:
			dest = Path(tmp) / "file.bin"
			part = dest.with_name(dest.name + ".part")
			part.write_bytes(b"already-down")
			rest = b"loaded-tail"
			full = b"already-down" + rest
			expected = hashlib.sha256(full).hexdigest()
			opener = mock.Mock(return_value=FakeResponse(rest, status=206))
			setup.download_file("http://example.test/file.bin", dest, expected, urlopen=opener)
			request = opener.call_args[0][0]
			self.assertEqual(request.get_header("Range"), "bytes=12-")
			self.assertEqual(dest.read_bytes(), full)

	def test_part_ignored_and_restarted_when_server_does_not_honour_range(self):
		with tempfile.TemporaryDirectory() as tmp:
			dest = Path(tmp) / "file.bin"
			part = dest.with_name(dest.name + ".part")
			part.write_bytes(b"stale-partial-data")
			full = b"fresh-full-body"
			expected = hashlib.sha256(full).hexdigest()
			opener = mock.Mock(return_value=FakeResponse(full, status=200))  # ignored the Range header
			setup.download_file("http://example.test/file.bin", dest, expected, urlopen=opener)
			self.assertEqual(dest.read_bytes(), full)


class ExtractZipTests(unittest.TestCase):
	def test_extract_zip_writes_members_into_dest_dir(self):
		import zipfile
		with tempfile.TemporaryDirectory() as tmp:
			zip_path = Path(tmp) / "a.zip"
			with zipfile.ZipFile(zip_path, "w") as archive:
				archive.writestr("llama-server.exe", b"stub-exe")
				archive.writestr("ggml.dll", b"stub-dll")
			dest_dir = Path(tmp) / "out"
			names = setup.extract_zip(zip_path, dest_dir)
			self.assertEqual(sorted(names), ["ggml.dll", "llama-server.exe"])
			self.assertEqual((dest_dir / "llama-server.exe").read_bytes(), b"stub-exe")


class ConfigTests(unittest.TestCase):
	def test_build_config_sets_paths_and_default_server_args(self):
		config = setup.build_config({}, server_exe="tools/llama/llama-server.exe", model_path="models/m.gguf")
		self.assertEqual(config["server_exe"], "tools/llama/llama-server.exe")
		self.assertEqual(config["model_path"], "models/m.gguf")
		self.assertEqual(config["server_args"], setup.DEFAULT_SERVER_ARGS)

	def test_build_config_keeps_existing_server_args_when_not_overridden(self):
		existing = {"server_args": "-c 4096", "some_other_key": "keep-me"}
		config = setup.build_config(existing, server_exe="e", model_path="m")
		self.assertEqual(config["server_args"], "-c 4096")
		self.assertEqual(config["some_other_key"], "keep-me")

	def test_build_config_overrides_server_args_when_given(self):
		existing = {"server_args": "-c 4096"}
		config = setup.build_config(existing, server_exe="e", model_path="m", server_args="--new-flag")
		self.assertEqual(config["server_args"], "--new-flag")

	def test_write_and_load_config_round_trip(self):
		with tempfile.TemporaryDirectory() as tmp:
			path = Path(tmp) / "local_llm_server.json"
			setup.write_config(path, {"server_exe": "e", "model_path": "m"})
			self.assertEqual(setup.load_existing_config(path), {"server_exe": "e", "model_path": "m"})

	def test_load_existing_config_missing_or_corrupt_returns_empty(self):
		with tempfile.TemporaryDirectory() as tmp:
			path = Path(tmp) / "missing.json"
			self.assertEqual(setup.load_existing_config(path), {})
			path.write_text("{not json", encoding="utf-8")
			self.assertEqual(setup.load_existing_config(path), {})


class CliArgvTests(unittest.TestCase):
	def test_default_cuda_is_13_4_and_pins_known_assets(self):
		args = setup.parse_args([])
		self.assertEqual(args.cuda, "13.4")
		names = [name for name, _ in setup.ASSETS_BY_CUDA[args.cuda]]
		self.assertEqual(names, ["llama-b11200-bin-win-cuda-13.4-x64.zip", "cudart-llama-bin-win-cuda-13.4-x64.zip"])

	def test_cuda_12_4_option_selects_the_other_pinned_pair(self):
		args = setup.parse_args(["--cuda", "12.4"])
		names = [name for name, _ in setup.ASSETS_BY_CUDA[args.cuda]]
		self.assertEqual(names, ["llama-b11200-bin-win-cuda-12.4-x64.zip", "cudart-llama-bin-win-cuda-12.4-x64.zip"])

	def test_model_url_and_sha256_are_overridable(self):
		args = setup.parse_args(["--model-url", "http://x/y.gguf", "--model-sha256", "abc"])
		self.assertEqual((args.model_url, args.model_sha256), ("http://x/y.gguf", "abc"))

	def test_pinned_asset_sha256_values_are_64_hex_chars(self):
		for assets in setup.ASSETS_BY_CUDA.values():
			for name, sha256 in assets:
				with self.subTest(name=name):
					self.assertEqual(len(sha256), 64)
					int(sha256, 16)  # raises ValueError if not hex


class MainOrchestrationTests(unittest.TestCase):
	def test_main_downloads_pinned_assets_and_model_then_writes_merged_config(self):
		with tempfile.TemporaryDirectory() as tmp:
			program_dir = Path(tmp)
			(program_dir / "local_llm_server.json").write_text(
				json.dumps({"server_args": "-c 4096 --n-cpu-moe 26"}), encoding="utf-8",
			)
			download_calls = []
			extract_calls = []

			def fake_download(url, dest, sha256, **_kwargs):
				download_calls.append((url, dest, sha256))
				dest.parent.mkdir(parents=True, exist_ok=True)
				dest.write_bytes(b"stub")

			def fake_extract(zip_path, dest_dir):
				extract_calls.append((zip_path, dest_dir))
				dest_dir.mkdir(parents=True, exist_ok=True)
				(dest_dir / "llama-server.exe").write_bytes(b"stub-exe")
				return ["llama-server.exe"]

			with mock.patch.object(setup, "download_file", side_effect=fake_download), \
					mock.patch.object(setup, "extract_zip", side_effect=fake_extract):
				code = setup.main(["--dest", str(program_dir)])

			self.assertEqual(code, 0)
			self.assertEqual(len(download_calls), 3)  # server zip + cudart zip + model
			self.assertEqual(len(extract_calls), 2)
			config = json.loads((program_dir / "local_llm_server.json").read_text(encoding="utf-8"))
			self.assertEqual(config["server_exe"], "tools/llama/llama-server.exe")
			self.assertEqual(config["model_path"], f"models/gemma-4-26B_q4_0-it.gguf")
			# a user's own tuning in the pre-existing file must survive an unrelated rerun
			self.assertEqual(config["server_args"], "-c 4096 --n-cpu-moe 26")

	def test_main_server_args_option_overrides_existing_config(self):
		with tempfile.TemporaryDirectory() as tmp:
			program_dir = Path(tmp)
			with mock.patch.object(setup, "download_file"), \
					mock.patch.object(setup, "extract_zip", return_value=["llama-server.exe"]):
				(program_dir / "tools" / "llama").mkdir(parents=True)
				(program_dir / "tools" / "llama" / "llama-server.exe").write_bytes(b"stub-exe")
				code = setup.main(["--dest", str(program_dir), "--server-args", "--n-cpu-moe 26"])
			self.assertEqual(code, 0)
			config = json.loads((program_dir / "local_llm_server.json").read_text(encoding="utf-8"))
			self.assertEqual(config["server_args"], "--n-cpu-moe 26")

	def test_main_rejects_malformed_server_args_before_downloading_anything(self):
		with tempfile.TemporaryDirectory() as tmp:
			program_dir = Path(tmp)
			with mock.patch.object(setup, "download_file") as download, \
					self.assertRaises(Exception):
				setup.main(["--dest", str(program_dir), "--server-args", '"unterminated'])
			download.assert_not_called()


if __name__ == "__main__":
	unittest.main()
