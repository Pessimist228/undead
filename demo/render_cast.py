#!/usr/bin/env python3
"""Render an asciicast v2 file to MP4 and GIF.

    python demo/render_cast.py demo/demo.cast demo/demo.mp4 --gif demo/demo.gif

Needs pyte, Pillow and ffmpeg. The terminal is emulated with pyte and drawn with a
monospace font: the text in the video is the text the recorded terminal showed.

For the demo recording the renderer adds a layer of annotations on top of that text:

  - time. The moment of the kill plays at quarter speed, the control run at double
    speed, and idle stretches are shortened. A badge shows the speed whenever it is
    not 1x.
  - a red mark where the stream stopped, and a blue tint on the text that arrived
    after the resume.
  - red on the kill lines of the right pane and green on the two hashes at the end.
  - one caption line under the terminal.

The moments are found by the strings demo/client.py and demo/kill_serving.sh print.
If they are not in the recording, or with --plain, the recording is rendered as it
is, in real time and without annotations.
"""
import argparse
import bisect
import json
import re
import subprocess
import sys

import pyte
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
FONT_UI = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_UI_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
OUTER, BG, FG = (1, 4, 9), (13, 17, 23), (201, 209, 217)
LINE, MUTED = (48, 54, 61), (139, 148, 158)
RED, GREEN, BLUE, AMBER = (255, 123, 114), (86, 211, 100), (121, 192, 255), (227, 179, 65)
RED_BG = (61, 20, 24)
COLORS = {
    "black": (72, 79, 88), "red": (255, 123, 114), "green": (63, 185, 80), "brown": (210, 153, 34),
    "blue": (88, 166, 255), "magenta": (188, 140, 255), "cyan": (57, 197, 207), "white": (201, 209, 217),
    "brightblack": (110, 118, 129), "brightred": (255, 161, 152), "brightgreen": (86, 211, 100),
    "brightbrown": (227, 179, 65), "brightblue": (121, 192, 255), "brightmagenta": (210, 168, 255),
    "brightcyan": (86, 212, 221), "brightwhite": (240, 246, 252),
}
SLOW, FAST = 0.25, 2.0  # speed at the kill and in the control run
RAMP_IN, RAMP_OUT = 0.45, 0.55  # seconds of recording over which the speed eases down and back up
IDLE = 3.0              # an idle stretch is shown for at most this long
HOLD = 4.5              # how long the last frame stays


def rgb(name, default):
    if name == "default":
        return default
    if name in COLORS:
        return COLORS[name]
    if len(name) == 6:
        try:
            return tuple(int(name[i:i + 2], 16) for i in (0, 2, 4))
        except ValueError:
            pass
    return default


def row_text(screen, y):
    """The row as a string with exactly one character per cell."""
    line = screen.buffer[y]
    return "".join((line[x].data or " ")[:1] for x in range(screen.columns))


def ease(x):
    """Smoothstep: 0 to 1 with no jerk at either end."""
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


def first(events, needle, after=0.0):
    for t, kind, data in events:
        if kind == "o" and t > after and needle in data:
            return t
    return None


class Plan:
    """What happens when in the recording, and how fast each part is shown."""

    def __init__(self, events, cols, rows, last):
        self.ok = False
        self.start, self.end = 0.0, last
        self.t_cmd = first(events, "client.py")
        self.t_kill = first(events, "kill -9")
        if self.t_cmd is None or self.t_kill is None:
            return
        self.t_done1 = first(events, "sha256", self.t_kill)
        self.t_clear = first(events, "clear;", self.t_done1 or 1e9)
        self.t_done2 = first(events, "identical", self.t_clear or 1e9)
        if None in (self.t_done1, self.t_clear, self.t_done2):
            return
        pause = [re.search(r"Pause: (\d+) ms", d) for t, k, d in events if k == "o"]
        self.pause_ms = next((m.group(1) for m in pause if m), None)

        # One pass over the recording: where the panes split, when the left pane moved
        # again after the kill, and at which cell.
        screen = pyte.Screen(cols, rows)
        stream = pyte.Stream(screen)
        self.split = cols
        self.t_resume = self.cut = None
        for t, kind, data in events:
            if kind != "o":
                continue
            before = None
            if t > self.t_kill:
                before = [row_text(screen, y)[: self.split] for y in range(rows)]
            stream.feed(data)
            if self.split == cols:
                for x in range(cols):
                    if sum(1 for y in range(rows) if row_text(screen, y)[x] == "│") > rows // 2:
                        self.split = x
                        break
            if before is not None:
                for y in range(rows):
                    a, b = before[y], row_text(screen, y)[: self.split]
                    if a != b:
                        x = next(i for i in range(len(b)) if a[i] != b[i])
                        self.t_resume, self.cut = t, (y, x)
                        break
            if self.t_resume is not None:
                break
        if self.t_resume is None or self.t_resume > self.t_done1:
            return
        self.start = max(0.0, self.t_cmd - 0.6)
        self.end = self.t_done2 + HOLD
        self.slow = (self.t_kill - 0.15, self.t_resume + 0.25)
        self.ok = True

    def speed(self, t):
        a, b = self.slow
        if a - RAMP_IN < t < b + RAMP_OUT:
            # The speed is eased in the log domain, so the slowdown feels even all the way.
            if t < a:
                return SLOW ** ease((t - (a - RAMP_IN)) / RAMP_IN)
            if t > b:
                return SLOW ** (1.0 - ease((t - b) / RAMP_OUT))
            return SLOW
        if self.t_done1 <= t < self.t_clear:
            return max(1.0, (self.t_clear - self.t_done1) / IDLE)
        if self.t_clear <= t < self.t_done2:
            return FAST
        return 1.0

    def badge(self, t):
        if self.speed(t) < 0.8:
            return f"slow motion  {SLOW:g}×", AMBER
        if self.t_clear <= t < self.t_done2:
            return f"fast forward  {FAST:g}×", MUTED
        return None

    def caption(self, t):
        if t < self.slow[0] - RAMP_IN:
            return "A client streams an answer through the router", MUTED
        if t < self.t_kill:
            return "Next: kill -9 on the replica that is serving this answer", AMBER
        if t < self.t_resume:
            return "kill -9. The stream stops", RED
        if t < self.t_clear:
            after = f" after {self.pause_ms} ms" if self.pause_ms else ""
            return f"The other replica carries on{after}", BLUE
        if t < self.t_done2:
            return "Control: the same request and seed, nothing killed", MUTED
        return "Same token IDs in both answers", GREEN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cast")
    ap.add_argument("mp4")
    ap.add_argument("--gif", default=None)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--gif-fps", type=float, default=12.5)
    ap.add_argument("--font-size", type=int, default=17)
    ap.add_argument("--tail", type=float, default=1.0,
                    help="seconds to hold the last frame of a plain render")
    ap.add_argument("--plain", action="store_true",
                    help="real time, no annotations")
    args = ap.parse_args()

    with open(args.cast, encoding="utf-8") as f:
        header = json.loads(f.readline())
        events = [json.loads(line) for line in f if line.strip()]
    # Drop what the terminal prints when the recorded session ends, so the video
    # stops on the last real frame instead of a cleared screen.
    last = events[-1][0] if events else 0.0
    for e in events:
        if e[1] == "o" and ("\x1b[?1049l" in e[2] or "[exited]" in e[2]):
            last = e[0] - 0.1
            events = [x for x in events if x[0] < last]
            break
    cols, rows = header["width"], header["height"]
    plan = Plan(events, cols, rows, last)
    notes = plan.ok and not args.plain
    if not notes:
        plan.start, plan.end = 0.0, last + args.tail

    font = ImageFont.truetype(FONT, args.font_size)
    bold = ImageFont.truetype(FONT_BOLD, args.font_size)
    ui = ImageFont.truetype(FONT_UI, args.font_size + 1)
    ui_bold = ImageFont.truetype(FONT_UI_BOLD, args.font_size - 2)
    cw = round(font.getlength("M"))
    ch = round(args.font_size * 1.35)
    pad, margin = 16, 14 if notes else 0
    strip = round(ch * 2.2) if notes else 0
    width = (cols * cw + 2 * pad + 2 * margin + 1) // 2 * 2
    height = (rows * ch + 2 * pad + 2 * margin + strip + 1) // 2 * 2
    x0, y0 = margin + pad, margin + pad

    # The time map: video time for every millisecond of the recording.
    step = 0.001
    real, video = [plan.start], [0.0]
    while real[-1] < plan.end:
        t = real[-1]
        real.append(t + step)
        video.append(video[-1] + step / (plan.speed(t) if notes else 1.0))
    total = video[-1]

    screen = pyte.Screen(cols, rows)
    stream = pyte.Stream(screen)
    tinted = set()  # cells of the left pane that arrived after the resume

    def left(y):
        return row_text(screen, y)[: plan.split]

    def draw(t):
        img = Image.new("RGB", (width, height), OUTER if notes else BG)
        d = ImageDraw.Draw(img)
        if notes:
            d.rounded_rectangle([margin, margin, width - margin - 1, height - margin - 1],
                                radius=12, fill=BG, outline=LINE)
        texts = [row_text(screen, y) for y in range(rows)]
        kill_rows, hash_rows = set(), {}
        if notes:
            for y in range(rows):
                right = texts[y][plan.split + 1:]
                if "kill -9" in right or "is dead" in right:
                    kill_rows.add(y)
            if t >= plan.t_done2:
                for y in range(rows):
                    m = re.search(r"(sha256 of token IDs:|interrupted answer:)\s+", left(y))
                    if m:
                        hash_rows[y] = m.end()
            if plan.t_kill <= t < plan.t_resume + 0.12:
                for y in kill_rows:
                    if "kill -9" in texts[y]:
                        d.rectangle([x0 + (plan.split + 1) * cw - 2, y0 + y * ch,
                                     width - margin - 2, y0 + (y + 1) * ch], fill=RED_BG)
        for y in range(rows):
            line = screen.buffer[y]
            for x in range(cols):
                c = line[x]
                fg, bg = rgb(c.fg, FG), rgb(c.bg, BG)
                if c.reverse:
                    fg, bg = bg, fg
                if notes:
                    if (y, x) in tinted:
                        fg = BLUE
                    elif y in kill_rows and x > plan.split:
                        fg = RED
                    elif y in hash_rows and hash_rows[y] <= x < plan.split:
                        fg = GREEN
                px, py = x0 + x * cw, y0 + y * ch
                if bg != BG:
                    d.rectangle([px, py, px + cw, py + ch], fill=bg)
                if c.data.strip():
                    d.text((px, py), c.data, font=bold if c.bold else font, fill=fg)
        if notes:
            if plan.t_kill <= t < plan.t_clear:
                y, x = plan.cut
                mx = x0 + x * cw - cw // 2
                d.rectangle([mx - 2, y0 + y * ch - 1, mx + 1, y0 + (y + 1) * ch - 2], fill=RED)
            top = height - margin - strip
            d.line([margin + 1, top, width - margin - 2, top], fill=LINE)
            text, color = plan.caption(t)
            cy = top + strip // 2
            d.ellipse([x0, cy - 5, x0 + 10, cy + 5], fill=color)
            d.text((x0 + 22, cy), text, font=ui, fill=FG, anchor="lm")
            badge = plan.badge(t)
            if badge:
                label, color = badge
                w = round(ui_bold.getlength(label)) + 28
                bx = width - margin - pad - w
                d.rounded_rectangle([bx, cy - 15, bx + w, cy + 15], radius=15, outline=color, width=2)
                d.text((bx + w // 2, cy), label, font=ui_bold, fill=color, anchor="mm")
        return img

    cmd = ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{width}x{height}", "-r", str(args.fps), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", args.mp4]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    i, frame, n, key = 0, None, 0, None
    while n / args.fps <= total:
        t = real[min(bisect.bisect_right(video, n / args.fps), len(real) - 1)]
        changed = frame is None
        while i < len(events) and events[i][0] <= t:
            et, kind, data = events[i]
            i += 1
            if kind != "o":
                continue
            track = notes and plan.t_resume <= et < plan.t_done1
            before = [left(y) for y in range(rows)] if track else None
            stream.feed(data)
            changed = True
            if track:
                for y in range(rows):
                    a, b = before[y], left(y)
                    if a != b:
                        tinted.update((y, x) for x in range(len(b)) if a[x] != b[x] and b[x].strip())
            if notes and et >= plan.t_clear:
                tinted.clear()
        if notes:
            state = (plan.caption(t), plan.badge(t), plan.t_kill <= t < plan.t_resume + 0.12,
                     plan.t_kill <= t < plan.t_clear, t >= plan.t_done2)
            changed = changed or state != key
            key = state
        if changed:
            frame = draw(t).tobytes()
        ff.stdin.write(frame)
        n += 1
    ff.stdin.close()
    ff.wait()
    print(f"{args.mp4}: {width}x{height}, {n} frames, {total:.1f} s"
          + (", annotated" if notes else ", plain"))
    if args.gif:
        palette = args.gif + ".palette.png"
        vf = f"fps={args.gif_fps:g},scale=960:-1:flags=lanczos"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", args.mp4, "-vf",
                        vf + ",palettegen=max_colors=256", palette], check=True)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", args.mp4, "-i", palette,
                        "-lavfi", vf + " [x]; [x][1:v] paletteuse=dither=none", args.gif], check=True)
        subprocess.run(["rm", "-f", palette])
        print(f"{args.gif}: done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
