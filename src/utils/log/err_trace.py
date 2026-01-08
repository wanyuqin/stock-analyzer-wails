import os
import sys
import sysconfig
import traceback

def extract_core_stack(lines_num: int = 5) -> list:
    exc_type, exc_value, exc_tb = sys.exc_info()
    if exc_tb is None:
        return ["当前没有异常上下文"]

    frames = traceback.extract_tb(exc_tb)

    stdlib_path = sysconfig.get_paths().get("stdlib")
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    default_stdlib_hint = os.path.join(sys.base_prefix, "lib", f"python{py_ver}")

    noise_substrings = [
        os.sep + "site-packages" + os.sep,
        os.sep + "dist-packages" + os.sep,
        os.sep + "importlib" + os.sep,
        os.sep + "logging" + os.sep,
        os.sep + "asyncio" + os.sep,
        os.sep + "concurrent" + os.sep + "futures",
        os.sep + "multiprocessing" + os.sep,
        os.sep + "pkgutil.py",
        os.sep + "runpy.py",
        os.sep + "selectors.py",
        os.sep + "threading.py",
        os.sep + "contextlib.py",
        os.sep + "utils" + os.sep + "log" + os.sep + "err_trace.py",
    ]

    def is_noise(filename: str, name: str) -> bool:
        fn = os.path.normpath(filename)
        if stdlib_path and fn.startswith(os.path.normpath(stdlib_path)):
            return True
        if default_stdlib_hint and fn.startswith(os.path.normpath(default_stdlib_hint)):
            return True
        for s in noise_substrings:
            if s in fn:
                return True
        if name in {"__call__", "<module>", "wrapper"}:
            return True
        return False

    def short_path(path: str, max_parts: int = 3) -> str:
        try:
            rel = os.path.relpath(path, os.getcwd())
        except Exception:
            rel = path
        parts = rel.split(os.sep)
        if len(parts) > max_parts:
            return os.sep.join(parts[-max_parts:])
        return rel

    filtered = []
    for fr in frames:
        name = getattr(fr, "name", "") or ""
        if not is_noise(getattr(fr, "filename", ""), name):
            filtered.append(fr)

    if not filtered:
        filtered = frames

    if lines_num and lines_num > 0:
        filtered = filtered[-lines_num:]

    lines_out = ["Traceback (most recent call last):"]
    for fr in filtered:
        name = getattr(fr, "name", "") or ""
        base = f'File "{short_path(fr.filename)}", line {fr.lineno}'
        if name.strip():
            base += f", in {name}"
        lines_out.append(base)
        if getattr(fr, "line", None):
            lines_out.append(f"    {fr.line.strip()}")

    if exc_type:
        try:
            exc_only = traceback.format_exception_only(exc_type, exc_value)
            for ln in exc_only:
                lines_out.append(ln.rstrip("\n"))
        except Exception:
            exc_name = getattr(exc_type, "__name__", str(exc_type))
            exc_msg = str(exc_value) if exc_value is not None else ""
            lines_out.append(f"{exc_name}: {exc_msg}".rstrip())

    return lines_out