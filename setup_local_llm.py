"""下載本機語言模型服務所需的檔案，並設定自動啟動：

- llama.cpp Windows CUDA 版 llama-server（伺服器 zip + cudart zip），解壓到 tools/llama/
- 預設模型（Gemma 4 26B-A4B QAT q4_0），下載到 models/
- 兩者都會驗證 sha256；已經驗證通過的檔案會跳過下載
- 寫入 local_llm_server.json，讓程式下次啟動本機語言模型時自動啟動 llama-server

只有這支腳本會下載東西；主程式（live_caption_gemini.py／llama_server.py）本身從不下載。
只用標準函式庫，執行一次即可（重跑會跳過已經驗證過的檔案）。
"""

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from llama_server import DEFAULT_SERVER_ARGS, parse_server_args


# 釘住的 llama.cpp release：b11200（2026-09-26 從 GitHub API 取得的 asset digest）。
LLAMA_CPP_TAG = "b11200"
RELEASE_BASE_URL = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_CPP_TAG}"

# {cuda 版本: [(asset 檔名, sha256), ...]}；每個版本都是「server zip + cudart zip」一組。
ASSETS_BY_CUDA = {
	"13.4": [
		("llama-b11200-bin-win-cuda-13.4-x64.zip", "ac88b6102fb9cb6344f897ddfa7400e67ff8d687ad34c124ca5764293ef5ef3f"),
		("cudart-llama-bin-win-cuda-13.4-x64.zip", "738f8c251ac22b70c3ae6f83a10cf222725df0395246a2cf58f32bdb85fbe668"),
	],
	"12.4": [
		("llama-b11200-bin-win-cuda-12.4-x64.zip", "8f9fcdb185dcd99a63cdfa3716f1ab12584060baef96a48bc6db157a4db843d1"),
		("cudart-llama-bin-win-cuda-12.4-x64.zip", "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6"),
	],
}

DEFAULT_MODEL_URL = "https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-gguf/resolve/main/gemma-4-26B_q4_0-it.gguf"
DEFAULT_MODEL_SHA256 = "3eca3b8f6d7baf218a7dd6bba5fb59a56ee25fe2d567b6f5f589b4f697eca51d"

_CHUNK_SIZE = 1 << 20  # 1 MiB
# 沒有 timeout 的話，連線卡住（0 B/s）會永遠掛在 urlopen/read() 上，實測在 22MB 處卡死過。
DOWNLOAD_TIMEOUT = 60.0
MAX_DOWNLOAD_ATTEMPTS = 5
_RETRY_BACKOFF_SECONDS = 2.0


# ---------- sha256 / 下載（.part -> 驗證 -> rename，中斷後可以直接重跑） ----------

def sha256_of_file(path: Path, chunk_size: int = _CHUNK_SIZE) -> str:
	digest = hashlib.sha256()
	with open(path, "rb") as f:
		for chunk in iter(lambda: f.read(chunk_size), b""):
			digest.update(chunk)
	return digest.hexdigest()


def verify_existing(path: Path, expected_sha256: str) -> bool:
	"""True when ``path`` already exists and matches (safe to skip the download)."""
	return path.is_file() and sha256_of_file(path) == expected_sha256


def _content_length(response) -> int:
	try:
		return int(response.headers.get("Content-Length", 0))
	except (TypeError, ValueError):
		return 0


def download_file(
	url: str, dest: Path, expected_sha256: str, *, urlopen=urllib.request.urlopen,
	chunk_size: int = _CHUNK_SIZE, on_progress=None, timeout: float = DOWNLOAD_TIMEOUT,
	max_attempts: int = MAX_DOWNLOAD_ATTEMPTS, sleep=time.sleep,
) -> None:
	"""Stream ``url`` into ``dest`` via a ``.part`` file; only renamed after sha256 matches.

	A previous ``.part`` is resumed with a Range request when the server honours it (HTTP
	206); otherwise the download restarts from scratch. Interrupting this function (Ctrl+C,
	crash) never leaves a corrupt file at ``dest`` -- only ``.part`` is ever partial.

	``timeout`` bounds both the connect and every individual read -- without it a stalled
	connection (0 B/s) blocks forever instead of failing (hit this for real: hung at 22 MB
	against the GitHub release CDN). A stall or dropped connection is retried up to
	``max_attempts`` times with a short backoff, resuming from ``.part`` each time; on final
	failure ``.part`` is kept (never deleted) so simply re-running the script resumes it.

	A ``.part`` that already verifies (a previous run finished writing it but crashed
	before the rename) is used immediately, without any network request. HTTP 416 (the
	server saying our resume offset is already past the end -- an unverifiable, stale
	``.part``) drops it and restarts fresh instead of repeating the same failing Range
	request forever. HTTP 401/403/404 fail immediately: no retry will ever fix those.
	"""
	dest.parent.mkdir(parents=True, exist_ok=True)
	if verify_existing(dest, expected_sha256):
		if on_progress:
			on_progress(dest.name, "skip", 0, 0)
		return
	part = dest.with_name(dest.name + ".part")
	if part.is_file() and sha256_of_file(part) == expected_sha256:
		part.replace(dest)
		if on_progress:
			on_progress(dest.name, "skip", 0, 0)
		return

	last_error = None
	for attempt in range(1, max_attempts + 1):
		resume_from = part.stat().st_size if part.exists() else 0
		request = urllib.request.Request(url)
		if resume_from:
			request.add_header("Range", f"bytes={resume_from}-")
		try:
			response = urlopen(request, timeout=timeout)
		except urllib.error.HTTPError as error:
			if error.code in (401, 403, 404):
				raise RuntimeError(
					f"下載失敗（HTTP {error.code}）：{url}\n"
					"這類錯誤重試也不會成功，請確認網址是否正確、是否需要授權。"
				) from error
			if error.code == 416 and part.exists():
				# .part didn't verify above, and the server now says our resume offset is
				# at or past its end -- it's stale/corrupt, not something we can resume.
				part.unlink(missing_ok=True)
			last_error = error
		except urllib.error.URLError as error:
			last_error = error
		else:
			try:
				with response:
					resumed = resume_from and getattr(response, "status", 200) == 206
					mode = "ab" if resumed else "wb"
					written = resume_from if resumed else 0
					total = written + _content_length(response)
					with open(part, mode) as f:
						while True:
							chunk = response.read(chunk_size)
							if not chunk:
								break
							f.write(chunk)
							written += len(chunk)
							if on_progress:
								on_progress(dest.name, "downloading", written, total)
			except OSError as error:  # covers socket.timeout (== TimeoutError) and dropped connections
				last_error = error
			else:
				actual = sha256_of_file(part)
				if actual != expected_sha256:
					part.unlink(missing_ok=True)
					raise RuntimeError(f"下載內容 sha256 不符：{dest.name}（預期 {expected_sha256}，實際 {actual}），已刪除，請重新執行。")
				part.replace(dest)
				return
		if attempt < max_attempts:
			if on_progress:
				on_progress(dest.name, "retry", attempt, max_attempts)
			sleep(_RETRY_BACKOFF_SECONDS)

	raise RuntimeError(
		f"下載中斷（已重試 {max_attempts} 次仍失敗）：{dest.name}（{last_error}）\n"
		f"已下載的部分保留在 {part}，直接重新執行這支腳本就會從中斷處繼續下載。"
	)


def extract_zip(zip_path: Path, dest_dir: Path) -> list:
	"""Extract every member of ``zip_path`` directly into ``dest_dir``; return the names."""
	dest_dir.mkdir(parents=True, exist_ok=True)
	with zipfile.ZipFile(zip_path) as archive:
		names = archive.namelist()
		archive.extractall(dest_dir)
	return names


# ---------- local_llm_server.json：合併寫入，不覆蓋使用者自己的 server_args ----------

def load_existing_config(path: Path) -> dict:
	if not path.is_file():
		return {}
	try:
		data = json.loads(path.read_text(encoding="utf-8"))
	except (OSError, json.JSONDecodeError):
		return {}
	return data if isinstance(data, dict) else {}


def build_config(existing: dict, *, server_exe: str, model_path: str, server_args: str | None = None) -> dict:
	"""Merge onto ``existing`` so a user's own server_args tuning survives a rerun."""
	config = dict(existing)
	config["server_exe"] = server_exe
	config["model_path"] = model_path
	if server_args is not None:
		config["server_args"] = server_args
	elif "server_args" not in config:
		config["server_args"] = DEFAULT_SERVER_ARGS
	return config


def write_config(path: Path, config: dict) -> None:
	path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def _relative_path_str(path: Path, program_dir: Path) -> str:
	try:
		return path.relative_to(program_dir).as_posix()
	except ValueError:
		return str(path)


# ---------- CLI ----------

def parse_args(argv=None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
	parser.add_argument("--cuda", choices=sorted(ASSETS_BY_CUDA), default="13.4", help="llama.cpp CUDA 版本（預設 13.4）")
	parser.add_argument("--model-url", default=DEFAULT_MODEL_URL, help="要下載的模型網址（預設 Gemma 4 26B-A4B QAT q4_0）")
	parser.add_argument("--model-sha256", default=DEFAULT_MODEL_SHA256, help="上面那個模型檔的 sha256（換模型時要一起換）")
	parser.add_argument("--model-filename", default=None, help="模型存檔檔名（預設從網址推斷）")
	parser.add_argument("--server-args", default=None, help="寫入 local_llm_server.json 的 server_args；不指定就保留原有設定或用預設值")
	parser.add_argument("--dest", default=None, help="程式資料夾（預設是這支腳本所在的資料夾）")
	args = parser.parse_args(argv)
	if args.model_url != DEFAULT_MODEL_URL and args.model_sha256 == DEFAULT_MODEL_SHA256:
		# 換了模型網址卻沿用預設模型的 sha256，驗證一定會失敗（而且是誤導人的失敗方式）；
		# 在這裡直接擋下來，比讓它跑到下載完才報 sha256 不符更清楚。
		parser.error("換了 --model-url 就必須同時指定對應的 --model-sha256，不能沿用預設模型的 sha256。")
	return args


def _print_progress(name: str, phase: str, written: int, total: int) -> None:
	if phase == "skip":
		print(f"  {name}：sha256 已符合，略過下載。")
		return
	if phase == "retry":
		# 這裡 written/total 被借用為「第幾次／最多幾次」，跟 downloading 階段的位元組數不同單位
		print(f"\n  {name}：下載中斷，{_RETRY_BACKOFF_SECONDS:g} 秒後重試（第 {written}/{total} 次）...")
		return
	mb = 1024 * 1024
	if total:
		percent = min(100, written * 100 // total)
		print(f"\r  {name}：{written / mb:.1f} / {total / mb:.1f} MB（{percent}%）", end="", flush=True)
		if written >= total:
			print()
	else:
		print(f"\r  {name}：{written / mb:.1f} MB", end="", flush=True)


def main(argv=None) -> int:
	args = parse_args(argv)
	if args.server_args is not None:
		parse_server_args(args.server_args)  # 提早驗證格式，設定壞掉的話現在就報錯

	program_dir = Path(args.dest).resolve() if args.dest else Path(__file__).resolve().parent
	tools_dir = program_dir / "tools" / "llama"
	models_dir = program_dir / "models"
	downloads_dir = program_dir / "downloads"

	print(f"llama.cpp {LLAMA_CPP_TAG}（CUDA {args.cuda}）...")
	for name, sha256 in ASSETS_BY_CUDA[args.cuda]:
		zip_path = downloads_dir / name
		download_file(f"{RELEASE_BASE_URL}/{name}", zip_path, sha256, on_progress=_print_progress)
		print(f"  解壓縮 {name} 到 {tools_dir} ...")
		extract_zip(zip_path, tools_dir)

	server_exe = tools_dir / "llama-server.exe"
	if not server_exe.is_file():
		print(f"警告：解壓縮後找不到 {server_exe}，請檢查下載的版本是否正確。")

	model_filename = args.model_filename or args.model_url.rsplit("/", 1)[-1]
	model_path = models_dir / model_filename
	print(f"模型 {model_filename} ...")
	download_file(args.model_url, model_path, args.model_sha256, on_progress=_print_progress)

	config_path = program_dir / "local_llm_server.json"
	existing = load_existing_config(config_path)
	config = build_config(
		existing,
		server_exe=_relative_path_str(server_exe, program_dir),
		model_path=_relative_path_str(model_path, program_dir),
		server_args=args.server_args,
	)
	write_config(config_path, config)
	print(f"已寫入設定檔：{config_path}")
	print("完成：下次啟動程式選「本機語言模型」時會自動啟動 llama-server（不需要再自己手動啟動）。")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
