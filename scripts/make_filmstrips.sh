#!/usr/bin/env bash
# Filmstrips for the mandatory video audit (gate G-R3).
#
# This gate exists because `amp_command7` (journal Run 12) passed every
# aggregate metric while CRAWLING, and only the video caught it. AGENTS.md
# therefore makes the video verdict outrank the metrics.
#
# Two strips per clip, and the distinction matters:
#   *_strip.png  1.5 fps, 6x4 tile  -- overview of the whole 20 s clip.
#   *_gait.png   8 fps over 2.5 s, centre-cropped and upscaled -- THE EVIDENCE.
#                At 1.5 fps a shuffle and a walk look identical; swing
#                clearance is only legible at 8 fps. Never judge gait off the
#                overview strip.
#
#   scripts/make_filmstrips.sh <out_dir> <prefix> <clip.mp4> [clip.mp4 ...]
#
# ffmpeg 7.0.2 at ~/.local/bin/ffmpeg (the system has none).
set -euo pipefail
FFMPEG="${FFMPEG:-$HOME/.local/bin/ffmpeg}"
OUT="${1:?usage: make_filmstrips.sh <out_dir> <prefix> <clip.mp4>...}"; shift
PREFIX="${1:?missing prefix}"; shift
GAIT_START="${GAIT_START:-6}"      # skip the drop-in settle at clip start
GAIT_DUR="${GAIT_DUR:-2.5}"
mkdir -p "$OUT"

for mp4 in "$@"; do
    [ -f "$mp4" ] || { echo "missing: $mp4" >&2; exit 1; }
    tag=$(basename "$mp4" .mp4)
    name="${PREFIX}_${tag}"
    cp -n "$mp4" "$OUT/$name.mp4" 2>/dev/null || true

    "$FFMPEG" -y -loglevel error -i "$mp4" \
        -vf "fps=1.5,scale=320:-1,tile=6x4" -frames:v 1 \
        "$OUT/${name}_strip.png"

    # Half width, centred; and the LOWER 70 % of the height. Measured, not
    # guessed: a symmetric iw/2 x ih/2 centre crop puts the trunk in frame and
    # cuts the FEET off, which destroys the one thing this strip exists to
    # show. The robot sits low and the ground plane is near the bottom edge,
    # so the crop has to be biased downward. Tiles are then normalised to
    # 300 px wide -> a ~1500x1352 sheet, matching the m2657 audit's size.
    # An upscale to 6400 px wide was tried and cost 9 MB per clip (82 MB
    # for five) while adding no legibility the crop had not already given.
    "$FFMPEG" -y -loglevel error -ss "$GAIT_START" -t "$GAIT_DUR" -i "$mp4" \
        -vf "fps=8,crop=iw/2:ih*0.7:iw/4:ih*0.3,scale=300:-1,tile=5x4" -frames:v 1 \
        "$OUT/${name}_gait.png"

    echo "$name: $(du -h "$OUT/${name}_gait.png" | cut -f1) gait, $(du -h "$OUT/${name}_strip.png" | cut -f1) strip"
done
