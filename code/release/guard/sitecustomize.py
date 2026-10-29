"""File-access guard of the public rebuild. rebuild_all.py puts this directory on PYTHONPATH, so Python imports this module
at start-up in every process the rebuild starts, worker processes included. It reads its configuration from the environment
variable P2_PUBLIC_GUARD (JSON: directories that may be read, directories that may be written, a log directory) and refuses
any other Python file access, any program other than this Python, and any Python network connection. Python file openings
are logged.
Without the variable the module does nothing.
"""
import os
import sys

_CFG = os.environ.get("P2_PUBLIC_GUARD")


def install(cfg):
    import atexit
    import json
    norm = lambda p: os.path.normcase(os.path.abspath(os.fsdecode(p)))
    read_roots = tuple(norm(p).rstrip("\\/") + os.sep for p in cfg["read"])
    read_files = frozenset(norm(p) for p in cfg.get("read_files", []))
    write_roots = tuple(norm(p).rstrip("\\/") + os.sep for p in cfg["write"])
    devices = {os.path.normcase(x) for x in (os.devnull, "nul", "NUL", "CONIN$", "CONOUT$", "/dev/null")}
    exe = norm(sys.executable)
    opened, busy = set(), [False]

    def allowed(path, write):
        if isinstance(path, int) or path is None:
            return True
        p = os.fsdecode(path)
        if os.path.normcase(p) in devices:
            return True
        q = norm(p) + os.sep
        return q.startswith(write_roots) or (not write and (q.startswith(read_roots) or norm(p) in read_files))

    def refuse(what, path):
        raise PermissionError(f"public rebuild: {what} outside this package and the Python and font directories: {os.fsdecode(path)}")

    def check_program(first, argv):
        if norm(first) != exe:
            refuse("program", first)

    def hook(event, args):
        if busy[0]:
            return
        busy[0] = True
        try:
            if event == "open":
                path, mode, flags = (list(args) + [None, None, None])[:3]
                if isinstance(path, (str, bytes, os.PathLike)):
                    write = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
                        mode is None and isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC))
                    if not allowed(path, bool(write)):
                        refuse("write" if write else "read", path)
                    opened.add((norm(path), "w" if write else "r"))
            elif event in ("os.listdir", "os.scandir", "os.chdir"):
                p = args[0] if args else None
                if not allowed("." if p is None else p, False):
                    refuse("listing", p)
            elif event == "glob.glob":
                pat = os.fsdecode(args[0])
                cut = min([i for i, c in enumerate(pat) if c in "*?["] + [len(pat)])
                if not allowed(os.path.dirname(pat[:cut]) or ".", False):
                    refuse("listing", pat)
            elif event in ("os.mkdir", "os.remove", "os.rmdir", "os.truncate", "os.utime", "os.chmod", "shutil.rmtree"):
                if args and not allowed(args[0], True):
                    refuse("change", args[0])
            elif event in ("os.rename", "shutil.copyfile", "shutil.copytree", "shutil.move", "os.link", "os.symlink"):
                src, dst = args[0], args[1]
                if not allowed(src, event in ("os.rename", "shutil.move")) or not allowed(dst, True):
                    refuse("copy or move", f"{os.fsdecode(src)} -> {os.fsdecode(dst)}")
            elif event == "subprocess.Popen":
                executable, argv = args[0], args[1]
                if executable is not None:
                    first = executable
                elif isinstance(argv, (list, tuple)):
                    first = argv[0] if argv else ""
                else:                                     # on Windows the command line arrives as one string
                    s = os.fsdecode(argv).strip()
                    first = s[1:].split('"', 1)[0] if s.startswith('"') else (s.split() or [""])[0]
                check_program(first, argv)
                if len(args) > 2 and args[2] is not None and not allowed(args[2], False):
                    refuse("working directory", args[2])
            elif event == "_winapi.CreateProcess":       # worker processes; subprocess.Popen has already checked its own
                app = args[0] if args else None
                if isinstance(app, (str, bytes)) and app:
                    check_program(app, args[1])
            elif event in ("os.system", "os.startfile", "os.exec", "os.spawn", "os.posix_spawn", "socket.connect", "socket.getaddrinfo",
                           "urllib.Request", "webbrowser.open"):
                refuse(event, str(args[0]) if args else event)
        finally:
            busy[0] = False

    def dump():
        log = cfg.get("log")
        if log:
            busy[0] = True
            try:                                          # modules imported before the guard (this one among them) count as read
                for m in list(sys.modules.values()):
                    f = getattr(m, "__file__", None)
                    if isinstance(f, str):
                        opened.add((norm(f), "r"))
                with open(os.path.join(log, f"opened_{os.getpid()}.json"), "w", encoding="utf-8") as f:
                    json.dump(dict(argv=sys.argv, opened=sorted(opened)), f)
            finally:
                busy[0] = False
    atexit.register(dump)
    sys.addaudithook(hook)


if _CFG:
    import json as _json
    install(_json.loads(_CFG))
