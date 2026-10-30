"""Standard-library-only process lifetime for the CMD fallback on Windows."""
import ctypes
from ctypes import wintypes
from contextlib import nullcontext
import os
from pathlib import Path
import subprocess
import sys
import time


class ProcessFamily:
    """The owner joins before spawning; closing its handle reaps descendants."""

    def __init__(self):
        class Basic(ctypes.Structure):
            _fields_ = [('process_time', ctypes.c_int64), ('job_time', ctypes.c_int64),
                        ('flags', wintypes.DWORD), ('minimum', ctypes.c_size_t),
                        ('maximum', ctypes.c_size_t), ('active_limit', wintypes.DWORD),
                        ('affinity', ctypes.c_size_t), ('priority', wintypes.DWORD),
                        ('scheduling', wintypes.DWORD)]

        class Limits(ctypes.Structure):
            _fields_ = [('basic', Basic), ('io', ctypes.c_uint64 * 6),
                        ('process_memory', ctypes.c_size_t), ('job_memory', ctypes.c_size_t),
                        ('peak_process_memory', ctypes.c_size_t), ('peak_job_memory', ctypes.c_size_t)]

        self.kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        signatures = {
            'CreateJobObjectW': ([ctypes.c_void_p, wintypes.LPCWSTR], wintypes.HANDLE),
            'SetInformationJobObject': ([wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD], wintypes.BOOL),
            'AssignProcessToJobObject': ([wintypes.HANDLE, wintypes.HANDLE], wintypes.BOOL),
            'GetCurrentProcess': ([], wintypes.HANDLE),
            'QueryInformationJobObject': ([wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p], wintypes.BOOL),
            'OpenProcess': ([wintypes.DWORD, wintypes.BOOL, wintypes.DWORD], wintypes.HANDLE),
            'IsProcessInJob': ([wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)], wintypes.BOOL),
            'TerminateProcess': ([wintypes.HANDLE, wintypes.UINT], wintypes.BOOL),
            'TerminateJobObject': ([wintypes.HANDLE, wintypes.UINT], wintypes.BOOL),
            'CloseHandle': ([wintypes.HANDLE], wintypes.BOOL),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(self.kernel, name)
            function.argtypes, function.restype = arguments, result
        self.limits = Limits
        self.handle = self.kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            self._limits(True)
            if not self.kernel.AssignProcessToJobObject(self.handle, self.kernel.GetCurrentProcess()):
                raise ctypes.WinError(ctypes.get_last_error())
        except BaseException:
            self.kernel.CloseHandle(self.handle)
            self.handle = None
            raise

    def _limits(self, kill):
        value = self.limits()
        value.basic.flags = 0x2000 if kill else 0
        if not self.kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(value), ctypes.sizeof(value)):
            raise ctypes.WinError(ctypes.get_last_error())

    def __enter__(self):
        return self

    def __exit__(self, *_):
        buffer = ctypes.create_string_buffer(1024 * 1024)
        for _ in range(10):
            if not self.kernel.QueryInformationJobObject(self.handle, 3, buffer, len(buffer), None):
                self.kernel.TerminateJobObject(self.handle, 1)
                return
            count = wintypes.DWORD.from_buffer(buffer, 4).value
            children = 0
            for index in range(count):
                pid = ctypes.c_size_t.from_buffer(buffer, 8 + index * ctypes.sizeof(ctypes.c_size_t)).value
                if pid == os.getpid():
                    continue
                process = self.kernel.OpenProcess(1 | 0x100000 | 0x1000, False, pid)
                if not process:
                    continue
                try:
                    member = wintypes.BOOL()
                    if self.kernel.IsProcessInJob(process, self.handle, ctypes.byref(member)) and member.value:
                        children += 1
                        self.kernel.TerminateProcess(process, 0)
                finally:
                    self.kernel.CloseHandle(process)
            if not children:
                self._limits(False)
                self.kernel.CloseHandle(self.handle)
                self.handle = None
                return
            time.sleep(.02)
        self.kernel.TerminateJobObject(self.handle, 1)


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    expected = Path(__file__).resolve().parents[1] / 'app.py'
    if not arguments or Path(arguments[0]).resolve() != expected or not expected.is_file():
        raise ValueError('CMD supervisor requires this package\'s app.py')
    print('Drama-3 T8 · 保持此窗口运行，关闭窗口将停止服务并释放模型内存。', flush=True)
    with ProcessFamily() if os.name == 'nt' else nullcontext():
        try:
            return subprocess.call([sys.executable, '-X', 'utf8', str(expected), *arguments[1:]], cwd=expected.parent)
        except KeyboardInterrupt:
            return 130


if __name__ == '__main__':
    raise SystemExit(main())
