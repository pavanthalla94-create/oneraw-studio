import subprocess
import sys
import os
import logging
import shutil
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def check_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise EnvironmentError(
            "ffmpeg not found in PATH. Install it from https://ffmpeg.org/download.html"
        )
    log.info(f"Using ffmpeg: {ffmpeg}")
    return ffmpeg


def validate_input(path: str) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")
    if not p.is_file():
        raise ValueError(f"Path is not a file: {p}")
    return p


def build_filter_chain() -> str:
    # Strip sub-80 Hz rumble → cap at 8 kHz for voice clarity →
    # reduce stationary noise via afftdn → normalize loudness with dynaudnorm.
    return (
        "highpass=f=80,"
        "lowpass=f=8000,"
        "afftdn=nf=-25,"
        "dynaudnorm=f=150:g=15"
    )


def enhance(input_path: str, output_path: str = "enhanced_video.mp4") -> None:
    ffmpeg = check_ffmpeg()
    src = validate_input(input_path)
    dst = Path(output_path)

    if dst.exists():
        log.warning(f"Output file already exists and will be overwritten: {dst}")

    audio_filter = build_filter_chain()

    cmd = [
        ffmpeg,
        "-y",                       # overwrite output without prompting
        "-i", str(src),             # input file
        "-af", audio_filter,        # apply audio filter chain
        "-c:v", "copy",             # copy video codec — no re-encode
        "-c:a", "aac",              # encode audio as AAC
        "-b:a", "192k",             # audio bitrate
        "-ar", "48000",             # sample rate 48 kHz
        "-movflags", "+faststart",  # web-optimised MP4 atom placement
        str(dst),
    ]

    log.info("Starting audio enhancement…")
    log.info(f"  Source  : {src}")
    log.info(f"  Output  : {dst}")
    log.info(f"  Filters : {audio_filter}")
    log.debug(f"  Command : {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as exc:
        raise EnvironmentError(f"Failed to launch ffmpeg: {exc}") from exc

    if result.returncode != 0:
        log.error("ffmpeg exited with errors:\n%s", result.stderr)
        raise RuntimeError(
            f"ffmpeg failed (exit code {result.returncode}). "
            "See error output above."
        )

    log.info("Enhancement complete.")
    log.info(f"Saved to: {dst.resolve()}")


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: python {Path(__file__).name} <input_video> [output_video]")
        print("  output_video defaults to enhanced_video.mp4")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) >= 3 else "enhanced_video.mp4"

    try:
        enhance(input_file, output_file)
    except (FileNotFoundError, ValueError, EnvironmentError) as exc:
        log.error(str(exc))
        sys.exit(1)
    except RuntimeError as exc:
        log.error(str(exc))
        sys.exit(2)


if __name__ == "__main__":
    main()
