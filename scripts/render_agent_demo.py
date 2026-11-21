"""Render the matched study's recorded Connect Four games to a captioned MP4."""

import argparse
from functools import lru_cache
import json
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw, ImageFont

from src.game.endgame import replay_board

WIDTH, HEIGHT, FPS = 1280, 720, 24
INK, MUTED, SURFACE = "#182737", "#4e6175", "#f2f6fa"
BLUE, BLUE_DARK, AMBER, AMBER_DARK = "#3374ca", "#164c99", "#e7a542", "#a25c0f"
GREEN, RED = "#207850", "#a33e37"
ROOT = Path(__file__).resolve().parents[1]


@lru_cache
def font(size, bold):
    paths = (
        Path("/System/Library/Fonts/Supplemental") / ("Verdana Bold.ttf" if bold else "Verdana.ttf"),
        Path("/usr/share/fonts/truetype/dejavu") / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"),
    )
    for path in paths:
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size=size)


def label(draw, xy, text, size=24, color=INK, bold=False, anchor=None):
    draw.text(xy, text, fill=color, font=font(size, bold), anchor=anchor)


def box(draw, pos, radius=18, fill="white", outline=None, width=1):
    draw.rounded_rectangle(pos, radius=radius, fill=fill, outline=outline, width=width)


def frame(title, eyebrow, detail=None):
    im = Image.new("RGB", (WIDTH, HEIGHT), SURFACE)
    draw = ImageDraw.Draw(im)
    label(draw, (62, 34), eyebrow, 17, BLUE_DARK, True)
    label(draw, (62, 65), title, 43, bold=True)
    if detail:
        label(draw, (63, 126), detail, 21, MUTED)
    draw.line((62, 656, WIDTH - 62, 656), fill="#cdd9e5", width=2)
    label(draw, (64, 675), "CM32032  /  FUNCTION-APPROXIMATED REINFORCEMENT LEARNING", 15, MUTED, True)
    return im, draw


def counter(draw, center, player, radius=22):
    x, y = center
    color, rim = (BLUE, BLUE_DARK) if player == 1 else (AMBER, AMBER_DARK)
    draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color, outline=rim, width=2)
    draw.arc((x-radius+4, y-radius+4, x+radius-4, y+radius-4), 205, 325, fill="white", width=2)


def board_panel(draw, x, title, guide, board, marker=None, falling=None, outcome=None):
    y = 170
    box(draw, (x, y, x+561, 620), radius=20, outline="#d8e2ed")
    label(draw, (x+24, y+18), title, 26, bold=True)
    label(draw, (x+24, y+53), guide, 18, MUTED)
    stride, left, top = 49, x + 133, y + 125
    box(draw, (left-28, top-28, left+6*stride+28, top+5*stride+28),
        radius=14, fill="#dae7f5", outline="#c7daed", width=2)
    for col in range(7):
        label(draw, (left+col*stride, top-40), str(col+1), 16, MUTED, anchor="mm")
    state = board.encode()
    for row in range(6):
        for col in range(7):
            cx, cy = left+col*stride, top+(5-row)*stride
            if state[0, row, col] or state[1, row, col]:
                player = board.current_player if state[0, row, col] else 3-board.current_player
                counter(draw, (cx, cy), player)
            else:
                draw.ellipse((cx-21, cy-21, cx+21, cy+21), fill="#f9fbfe", outline="#c1d1e1", width=2)
    if marker is not None:
        col, tone = marker
        cx = left + col*stride
        draw.rounded_rectangle((cx-24, top-26, cx+24, top+5*stride+25), 12, outline=tone, width=4)
    if falling is not None:
        col, progress = falling
        row = int(state[:2, :, col].sum())
        cx, target_y = left+col*stride, top+(5-row)*stride
        start_y = top-10
        counter(draw, (cx, round(start_y + (target_y-start_y)*progress**1.6)), board.current_player)
    if outcome:
        tone = GREEN if outcome == "Agent wins" else RED
        box(draw, (x+24, 580, x+191, 609), 13, fill="#eaf2fa")
        label(draw, (x+34, 584), outcome, 17, tone, True)


def get_game(directory, name, pair):
    payload = json.loads((directory / name).read_text())
    return next(row for row in payload["games"] if row["pair"] == pair and row["player"] == 1)


def duration_frames(seconds):
    return int(round(seconds * FPS))


def intro():
    im, draw = frame("Learning to play Connect Four", "AGENT PERFORMANCE")
    label(draw, (64, 205), "One opening. Two checkpoints.", 36, bold=True)
    label(draw, (64, 265), "Compare the greedy Double DQN policy before and after self-play.", 25, MUTED)
    label(draw, (64, 324), "Then replay a tactical mistake against a stronger opponent.", 25, MUTED)
    box(draw, (62, 477, 1218, 584), fill="white", outline="#d8e2ed")
    counter(draw, (102, 521), 1, 16)
    label(draw, (128, 509), "Our agent", 22, bold=True)
    counter(draw, (302, 521), 2, 16)
    label(draw, (328, 509), "Opponent", 22, bold=True)
    label(draw, (75, 552), "Selected games from the held-out evaluation. Aggregate results follow.", 19, MUTED)
    return im


def closing(study):
    final = study["final"]
    im, draw = frame("The policy learned. Tactics still matter.", "HELD-OUT EVALUATION",
                     "Double DQN, three training seeds, 64 openings per seed with both seats")
    entries = [
        ("Against uniform random", final["random"]["ddqn"]["initial"]["mean"],
         final["random"]["ddqn"]["trained"]["mean"], "Greedy network before and after 5,000 games"),
        ("Against depth-two minimax", final["minimax"]["ddqn"]["initial"]["mean"],
         final["minimax"]["ddqn"]["trained"]["mean"], "Greedy network before and after 5,000 games"),
        ("Against depth-two minimax", final["minimax"]["ddqn"]["trained"]["mean"],
         final["minimax"]["ddqn"]["trained_guarded"]["mean"], "Same trained weights with the tactical filter"),
    ]
    for i, (title, before, after, reason) in enumerate(entries):
        y = 181 + i*147
        box(draw, (63, y, 1217, y+124), fill="white", outline="#d8e2ed")
        label(draw, (87, y+21), title, 24, bold=True)
        label(draw, (88, y+60), reason, 18, MUTED)
        label(draw, (1185, y+45), f"{before:.1%} to {after:.1%}", 34, BLUE_DARK, True, "rm")
    return im


def match_frames(info, pair, offset, mark_then_play):
    title, eyebrow, detail, panels = info
    boards = [replay_board(row["opening"] + row["moves_after_opening"][:offset]) for row in pair]
    continuations = [row["moves_after_opening"][offset:] for row in pair]

    def picture(falling=(None, None), marks=(None, None)):
        im, draw = frame(title, eyebrow, detail)
        for i in range(2):
            outcome = None
            if boards[i].is_terminal():
                outcome = "Agent wins" if boards[i].get_result(1) == 1 else "Agent loses"
            board_panel(draw, (62, 657)[i], panels[i][0], panels[i][1], boards[i],
                        marks[i], falling[i], outcome)
        return im

    marks = (None, None)
    if mark_then_play:
        marks = ((continuations[0][0], RED), (continuations[1][0], GREEN))
    yield picture(marks=marks), duration_frames(7.5 if mark_then_play else 4.5)
    for tick in range(max(map(len, continuations))):
        actions = [moves[tick] if tick < len(moves) else None for moves in continuations]
        for t in range(1, 7):
            falling = [None if action is None else (action, t/6) for action in actions]
            yield picture(falling=falling, marks=marks if tick == 0 else (None, None)), 1
        for i, action in enumerate(actions):
            if action is not None:
                boards[i] = boards[i].make_move(action)
        dwell = 2.4 if mark_then_play and tick == 1 else (0.48 if mark_then_play else 0.82)
        yield picture(), duration_frames(dwell)
    yield picture(), duration_frames(4.8)


def render(args):
    directory = args.study / "evaluation/ddqn-51"
    pair_one = (get_game(directory, "final-network-0-random.json", 2),
                get_game(directory, "final-network-5000-random.json", 2))
    pair_two = (get_game(directory, "final-network-5000-minimax.json", 7),
                get_game(directory, "final-guarded-5000-minimax.json", 7))
    for pair in (pair_one, pair_two):
        assert pair[0]["opening"] == pair[1]["opening"]
        assert [replay_board(p["opening"] + p["moves_after_opening"]).get_result(1)
                for p in pair] == [-1, 1]
    offset = 12
    assert pair_two[0]["moves_after_opening"][:offset] == pair_two[1]["moves_after_opening"][:offset]
    assert [p["moves_after_opening"][offset] for p in pair_two] == [5, 6]
    learning = ("The same opening, before and after", "01  /  LEARNING FROM FRESH WEIGHTS",
                "Uniform random opponent  •  Same opening, seat and random seed",
                (("Untrained network", "0 self-play games"), ("Trained network", "5,000 self-play games")))
    guard = ("An immediate loss the network misses", "02  /  WHAT THE SAFEGUARD ADDS",
             "Depth-two minimax  •  Opponent can win next in column 7",
             (("Greedy Q-values", "Column 6 leaves the threat open"),
              ("Greedy Q-values + tactical filter", "Column 7 is the only move that blocks")))
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
               "-pixel_format", "rgb24", "-video_size", f"{WIDTH}x{HEIGHT}", "-framerate", str(FPS),
               "-i", "-", "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "19",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.output)]
    count = 0
    with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
        def write(frame_image, repeats):
            nonlocal count
            payload = frame_image.tobytes()
            for _ in range(repeats):
                encoder.stdin.write(payload)
            count += repeats
        write(intro(), duration_frames(6.0))
        for image, repeats in match_frames(learning, pair_one, 0, False):
            write(image, repeats)
        for image, repeats in match_frames(guard, pair_two, offset, True):
            write(image, repeats)
        write(closing(json.loads((args.study / "summary.json").read_text())), duration_frames(10.0))
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError("Video encoder failed")
    assert 0 < count / FPS < 120
    print(f"{count / FPS:.2f} seconds at {FPS} frames per second")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=ROOT / "study-evidence/dqn-study")
    parser.add_argument("--output", type=Path, required=True)
    render(parser.parse_args())


if __name__ == "__main__":
    main()
