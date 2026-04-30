import sys
import shutil
import logging
import subprocess
import argparse
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Color grade presets — each is an ordered list of FFmpeg video filters
# that will be joined into a single -vf chain.
# ---------------------------------------------------------------------------

PRESETS = {
    # Clean, punchy baseline — subtle contrast lift, unity saturation
    "neutral": [
        "eq=saturation=1.0:contrast=1.05:brightness=0.0:gamma=1.0",
    ],

    # Golden-hour look — lift reds/yellows, pull blues, boost saturation
    "warm": [
        "eq=saturation=1.3:contrast=1.1:brightness=0.05:gamma=0.95",
        # shadows: push red +0.10, pull blue -0.12
        # midtones: push red +0.06, push green +0.02, pull blue -0.08
        # highlights: push red +0.04, pull blue -0.05
        "colorbalance=rs=0.10:gs=0.0:bs=-0.12"
        ":rm=0.06:gm=0.02:bm=-0.08"
        ":rh=0.04:gh=0.0:bh=-0.05",
    ],

    # Cinematic / dark — desaturated, crushed blacks, teal-blue shadows
    "moody": [
        "eq=saturation=0.8:contrast=1.2:brightness=-0.05:gamma=1.05",
        # shadows: teal push (green+blue), pull red
        # highlights: subtle cool pull
        "colorbalance=rs=-0.05:gs=0.04:bs=0.08"
        ":rm=-0.02:gm=0.0:bm=0.03"
        ":rh=-0.02:gh=0.0:bh=0.04",
    ],

    # Social-media pop — punchy saturation, lifted contrast, warm-neutral balance
    "vivid": [
        "eq=saturation=1.5:contrast=1.1:brightness=0.0:gamma=0.97",
        # slight warm nudge in midtones to stop vivid from going neon-cold
        "colorbalance=rs=0.04:gs=0.01:bs=-0.04"
        ":rm=0.03:gm=0.01:bm=-0.03"
        ":rh=0.02:gh=0.0:bh=-0.02",
    ],
}

PRESET_NAMES = list(PRESETS.keys())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        log.error("ffmpeg not found in PATH. Install from https://ffmpeg.org/download.html")
        sys.exit(1)
    return ffmpeg


def validate_input(path: str) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")
    if not p.is_file():
        raise ValueError(f"Path is not a file: {p}")
    return p


def build_vf(preset: str) -> str:
    filters = PRESETS[preset]
    return ",".join(filters)


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def colorgrade(src: Path, preset: str, out: Path) -> None:
    ffmpeg = check_ffmpeg()
    vf = build_vf(preset)

    log.info("=" * 60)
    log.info(f"Preset   : {preset}")
    log.info(f"Source   : {src}")
    log.info(f"Output   : {out}")
    log.info(f"Filters  : {vf}")
    log.info("=" * 60)

    if out.exists():
        log.warning(f"Output file already exists and will be overwritten: {out}")

    cmd = [
        ffmpeg, "-y",
        "-i", str(src),
        "-vf", vf,
        "-c:v", "libx264",
        "-crf", "18",           # near-lossless to preserve grade accuracy
        "-preset", "medium",
        "-c:a", "copy",         # audio unchanged — no need to re-encode
        "-movflags", "+faststart",
        str(out),
    ]

    log.info("Running FFmpeg...")
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        log.error("FFmpeg failed:\n%s", result.stderr[-4000:])
        raise RuntimeError(f"FFmpeg exited with code {result.returncode}.")

    size_mb = out.stat().st_size / (1024 * 1024)
    log.info(f"Done. Saved to: {out.resolve()}  ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply FFmpeg color grading presets to a video.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("video", help="Path to the input video file")
    parser.add_argument(
        "--preset", "-p",
        choices=PRESET_NAMES,
        default="neutral",
        help=(
            "Color grade preset (default: neutral)\n"
            "  neutral  — clean, subtle contrast lift\n"
            "  warm     — golden-hour tones, lifted reds\n"
            "  moody    — cinematic, teal shadows, crushed blacks\n"
            "  vivid    — punchy saturation, social-media pop"
        ),
    )
    parser.add_argument(
        "--output", "-o",
        default="graded_video.mp4",
        help="Output file path (default: graded_video.mp4)",
    )
    args = parser.parse_args()

    try:
        src = validate_input(args.video)
    except (FileNotFoundError, ValueError) as exc:
        log.error(str(exc))
        sys.exit(1)

    out = Path(args.output)

    try:
        colorgrade(src, args.preset, out)
    except RuntimeError as exc:
        log.error(str(exc))
        sys.exit(2)


if __name__ == "__main__":
    main()
