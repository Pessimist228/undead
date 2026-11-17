#!/usr/bin/env python3
"""Record a command running in a pseudo-terminal to an asciicast v2 file.

    python demo/rec.py OUT.cast COLS ROWS -- command [args...]

A small stand-in for `asciinema rec` that needs no terminal of its own.
"""
import codecs
import fcntl
import json
import os
import pty
import select
import struct
import sys
import termios
import time


def main(argv):
    out, cols, rows = argv[1], int(argv[2]), int(argv[3])
    cmd = argv[argv.index("--") + 1:]
    pid, fd = pty.fork()
    if pid == 0:
        fcntl.ioctl(0, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
        os.environ["TERM"] = "xterm-256color"
        os.execvp(cmd[0], cmd)
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    start = time.monotonic()
    with open(out, "w", encoding="utf-8") as f:
        f.write(json.dumps({"version": 2, "width": cols, "height": rows,
                            "timestamp": int(time.time()),
                            "env": {"TERM": "xterm-256color"}}) + "\n")
        while True:
            ready, _, _ = select.select([fd], [], [], 0.2)
            if ready:
                try:
                    data = os.read(fd, 65536)
                except OSError:
                    break
                if not data:
                    break
                text = decoder.decode(data)
                if text:
                    f.write(json.dumps([round(time.monotonic() - start, 4), "o", text]) + "\n")
                    f.flush()
            elif os.waitpid(pid, os.WNOHANG)[0]:
                break
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
