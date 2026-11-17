#!/usr/bin/env python3
"""Demo client: streams one answer through the router and prints what happened to it.

    python demo/client.py --seed 42 --save-hash /tmp/demo.sha
    python demo/client.py --seed 42 --compare-hash /tmp/demo.sha

Uses only the standard library. The summary comes from the router's debug endpoint:
which replicas served the answer, the pause, and the sha256 of the token IDs.
"""
import argparse
import http.client
import json
import os
import shutil
import sys
import time
from urllib.parse import urlparse

BOLD, DIM, GREEN, RED, RESET = "\033[1m", "\033[2m", "\033[32m", "\033[31m", "\033[0m"
PROMPT = "Write a short story about a lighthouse keeper who finds a note in a bottle."


class Wrapper:
    """Word-wraps a stream of text pieces to the terminal width."""

    def __init__(self, width):
        self.width, self.col, self.word = width, 0, ""

    def feed(self, text):
        for ch in text:
            if ch in " \n":
                self.flush()
                if ch == "\n":
                    sys.stdout.write("\n")
                    self.col = 0
                elif self.col:
                    sys.stdout.write(" ")
                    self.col += 1
            else:
                self.word += ch
        sys.stdout.flush()

    def flush(self):
        if not self.word:
            return
        if self.col + len(self.word) > self.width:
            sys.stdout.write("\n")
            self.col = 0
        sys.stdout.write(self.word)
        self.col += len(self.word)
        self.word = ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--router", default=os.environ.get("UNDEAD_ROUTER", "http://127.0.0.1:8090"))
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=340)
    ap.add_argument("--save-hash", default=None)
    ap.add_argument("--compare-hash", default=None)
    args = ap.parse_args()

    u = urlparse(args.router)
    width = shutil.get_terminal_size((78, 24)).columns - 1
    if args.compare_hash:
        print(f"{BOLD}CONTROL: the same request and seed again, nothing gets killed{RESET}")
    print(f"{DIM}seed {args.seed}, temperature {args.temperature}, max_tokens {args.max_tokens}{RESET}")
    print(f"{BOLD}> {args.prompt}{RESET}\n")

    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=300)
    body = {"messages": [{"role": "user", "content": args.prompt}], "stream": True,
            "seed": args.seed, "temperature": args.temperature, "max_tokens": args.max_tokens}
    conn.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    resp = conn.getresponse()
    if resp.status != 200:
        print(f"{RED}HTTP {resp.status}: {resp.read()[:300].decode(errors='replace')}{RESET}")
        return 1
    answer_id = resp.getheader("X-Answer-Id")
    wrap = Wrapper(width)
    last = time.monotonic()
    gap = 0.0
    error = None
    event = None
    for raw in resp:
        line = raw.decode("utf-8", errors="replace").rstrip("\n")
        if line.startswith("event:"):
            event = line[6:].strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        obj = json.loads(payload)
        if event == "error" or "error" in obj:
            error = obj.get("error", {}).get("message", "unknown error")
            break
        piece = obj["choices"][0]["delta"].get("content") or ""
        if piece:
            now = time.monotonic()
            gap = max(gap, now - last)
            last = now
            wrap.feed(piece)
    wrap.flush()
    conn.close()
    print("\n")
    if error:
        print(f"{RED}error: {error}{RESET}")
        return 1

    conn = http.client.HTTPConnection(u.hostname, u.port, timeout=30)
    conn.request("GET", f"/v1/answers/{answer_id}")
    info = json.loads(conn.getresponse().read())
    conn.close()
    replicas = " -> ".join(":" + r.rsplit(":", 1)[-1] for r in info["replicas"])
    sha = info["ids_sha256"]
    print(f"{BOLD}{info['tokens']} tokens, served by replica {replicas}{RESET}")
    if info["attempts"] > 1:
        pauses = ", ".join(f"{p * 1000:.0f} ms" for p in info["pauses"])
        print(f"replica died mid-answer, the router resumed it elsewhere. Pause: {pauses}")
    else:
        print("no failure during this answer")
    print(f"sha256 of token IDs: {sha[:32]}")
    if args.save_hash:
        with open(args.save_hash, "w") as f:
            f.write(sha)
    if args.compare_hash:
        with open(args.compare_hash) as f:
            other = f.read().strip()
        print(f"interrupted answer:  {other[:32]}")
        if other == sha:
            print(f"{GREEN}{BOLD}identical, token for token{RESET}")
        else:
            print(f"{RED}{BOLD}DIFFERENT{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
