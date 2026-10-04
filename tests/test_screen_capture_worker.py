"""screen_capture_worker.py when Windows Graphics Capture cannot be used: the window must fall back to PrintWindow.

windows_capture is swapped in sys.modules per test, so nothing here needs Windows or the real package.
"""

import importlib
import os
import re
import sys
import threading
import types
import unittest
from unittest import mock


def import_worker():
	# The worker only imports cv2/numpy here; real modules or another test's fakes both work.
	for name in ("cv2", "numpy"):
		try:
			importlib.import_module(name)
		except ImportError:
			stand_in = sys.modules[name] = types.ModuleType(name)
			stand_in.ndarray = object  # annotations are evaluated at import before Python 3.14
	return importlib.import_module("screen_capture_worker")


worker = import_worker()
HWND = 0x1234


class OldWindowsCapture:
	"""windows-capture 1.x signature: no window_hwnd parameter."""

	def __init__(self, cursor_capture=True, draw_border=None, secondary_window=None,
			minimum_update_interval=None, dirty_region=None, monitor_index=None, window_name=None):
		pass


class BrokenNativeCapture:
	def __init__(self, **kwargs):
		raise RuntimeError("synthetic native failure")


class FailingStartCapture:
	def __init__(self, **kwargs):
		self.kwargs = kwargs

	def event(self, handler):
		return handler

	def start_free_threaded(self):
		raise RuntimeError("synthetic start failure")


def fake_windows_capture(capture_class):
	module = types.ModuleType("windows_capture")
	module.WindowsCapture = capture_class
	return module


def broken_windows_capture():
	"""Installed package whose native submodule is gone, as a broken install leaves it."""
	module = types.ModuleType("windows_capture")

	def missing_submodule(name):
		if name == "WindowsCapture":
			raise ModuleNotFoundError(
				"No module named 'windows_capture.windows_capture'", name="windows_capture.windows_capture",
			)
		raise AttributeError(name)

	module.__getattr__ = missing_submodule
	return module


def required_windows_capture_version():
	path = os.path.join(os.path.dirname(worker.__file__), "requirements.txt")
	with open(path, encoding="utf-8") as f:
		match = re.search(r"^windows-capture>=([\d.]+)", f.read(), re.M)
	return match.group(1) if match else None


class StopAfter(threading.Event):
	"""Reports stop after a few checks, so a capture loop that never falls back cannot hang the suite."""

	def __init__(self, checks):
		super().__init__()
		self.checks_left = checks

	def is_set(self):
		self.checks_left -= 1
		return self.checks_left < 0 or super().is_set()


class WgcUnavailableTests(unittest.TestCase):
	def setUp(self):
		printer = mock.patch("builtins.print")
		self.printed = printer.start()
		self.addCleanup(printer.stop)

	def printed_text(self):
		return "\n".join(" ".join(str(arg) for arg in call.args) for call in self.printed.call_args_list)

	def run_wgc(self, windows_capture):
		with mock.patch.dict(sys.modules, {"windows_capture": windows_capture}):
			return worker._run_wgc(
				HWND, mock.Mock(name="writer"), types.SimpleNamespace(value=HWND),
				threading.Event(), threading.Event(),
			)

	def test_missing_package_returns_error_with_install_hint(self):
		self.assertEqual(self.run_wgc(None), "error")
		self.assertIn("windows-capture", self.printed_text())
		self.assertIn("pip install -r requirements.txt", self.printed_text())
		# The window stays on PrintWindow for the rest of the session, so installing alone is not enough
		self.assertIn("重新啟動程式", self.printed_text())

	def test_package_without_window_hwnd_returns_error_with_upgrade_hint(self):
		self.assertEqual(self.run_wgc(fake_windows_capture(OldWindowsCapture)), "error")
		self.assertIn("pip install -U windows-capture", self.printed_text())
		self.assertIn("重新啟動程式", self.printed_text())
		# Windows keeps the loaded 1.x .pyd locked, so upgrading while the program runs can fail
		self.assertIn("先關掉程式", self.printed_text())

	def test_upgrade_hint_names_the_minimum_version_in_requirements(self):
		minimum = required_windows_capture_version()
		self.assertIsNotNone(minimum, "requirements.txt has no windows-capture>= line")
		self.run_wgc(fake_windows_capture(OldWindowsCapture))
		self.assertIn(minimum, self.printed_text())

	def test_unloadable_package_returns_error_without_install_hint(self):
		# Without WindowsCapture the import raises a plain ImportError, not ModuleNotFoundError
		self.assertEqual(self.run_wgc(types.ModuleType("windows_capture")), "error")
		self.assertIn("無法使用 Windows Graphics Capture", self.printed_text())
		self.assertIn("cannot import name", self.printed_text())
		self.assertNotIn("pip install", self.printed_text())

	def test_broken_install_returns_error_without_install_hint(self):
		self.assertEqual(self.run_wgc(broken_windows_capture()), "error")
		self.assertIn("無法使用 Windows Graphics Capture", self.printed_text())
		self.assertIn("No module named 'windows_capture.windows_capture'", self.printed_text())
		self.assertNotIn("pip install", self.printed_text())

	def test_other_constructor_failure_returns_error(self):
		self.assertEqual(self.run_wgc(fake_windows_capture(BrokenNativeCapture)), "error")
		self.assertIn("synthetic native failure", self.printed_text())

	def test_start_failure_returns_error(self):
		self.assertEqual(self.run_wgc(fake_windows_capture(FailingStartCapture)), "error")
		self.assertIn("synthetic start failure", self.printed_text())

	def test_capture_process_switches_window_to_printwindow_when_package_is_missing(self):
		hwnd_shared = types.SimpleNamespace(value=HWND)
		printwindow_hwnds = []

		def fake_printwindow(hwnd, writer, hwnd_shared, stop_event, pause_event, target_fps):
			printwindow_hwnds.append(hwnd)
			stop_event.set()
			return "stop"

		# win32 modules hidden too, so the test runner's own priority is not lowered on Windows
		hidden = {"windows_capture": None, "win32api": None, "win32process": None}
		with mock.patch.dict(sys.modules, hidden), \
				mock.patch.object(worker, "_SlotWriter"), \
				mock.patch.object(worker, "_run_printwindow", fake_printwindow):
			worker.run_capture_process(
				hwnd_shared, 1280, 60, "shm", 0, None, None, StopAfter(checks=5), threading.Event(),
			)
		self.assertEqual(printwindow_hwnds, [HWND])


if __name__ == "__main__":
	unittest.main()
