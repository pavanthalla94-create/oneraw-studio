import sys
import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# Encoding — never stream copy
VIDEO_CODEC   = "libx264"
AUDIO_CODEC   = "aac"
CRF           = "23"
PRESET        = "medium"
AUDIO_BITRATE = "192k"
SAMPLE_RATE   = "48000"

REEL_W, REEL_H = 1080, 1920   # 9:16 vertical
YT_W,   YT_H   = 1920, 1080   # 16:9 horizontal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def check_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        log.error("ffmpeg not found in PATH.")
        sys.exit(1)
    return ffmpeg


def parse_ts(ts) -> float:
    """
    Convert to seconds. Accepts:
      float / int       → used as-is
      "0:20"            → MM:SS  → 20.0 s
      "1:05"            → MM:SS  → 65.0 s
      "00:01:05.500"    → HH:MM:SS.mmm → 65.5 s
    """
    if isinstance(ts, (int, float)):
        return float(ts)
    parts = str(ts).strip().split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(parts[0])
    except ValueError as exc:
        raise ValueError(f"Cannot parse timestamp '{ts}': {exc}") from exc


def get_time(clip: dict, key: str) -> float:
    """Read start or end from a clip dict; accept start/start_time variants."""
    for k in (key, key + "_time"):
        if k in clip:
            return parse_ts(clip[k])
    raise KeyError(f"Clip has no '{key}' or '{key}_time' field: {clip}")


# ---------------------------------------------------------------------------
# JSON loading
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def extract_sequences(data: dict) -> tuple:
    """
    Return (reel_sequence, youtube_sequence).
    Supports:
      suggested_sequence as a flat list → used for both
      suggested_sequence as dict with '90_second_reel' / '8_minute_youtube_video' keys
    """
    seq = data.get("suggested_sequence")
    if seq is None:
        raise ValueError("'suggested_sequence' key not found in JSON.")

    if isinstance(seq, list):
        return seq, seq

    if isinstance(seq, dict):
        reel_key = next((k for k in seq if "reel" in k.lower()), None)
        yt_key   = next((k for k in seq if "youtube" in k.lower() or "yt" in k.lower()), None)

        reel = seq.get(reel_key) if reel_key else None
        yt   = seq.get(yt_key)   if yt_key   else None

        if reel is None and yt is None:
            # fall back to first key for both
            first = next(iter(seq.values()))
            log.warning("No 'reel' or 'youtube' key found; using first sequence for both.")
            return first, first

        reel = reel or yt
        yt   = yt   or reel
        return reel, yt

    raise ValueError("'suggested_sequence' must be a list or object.")


def hook_start(data: dict):
    """Return hook start time in seconds, or None."""
    hook = data.get("hook")
    if not hook:
        return None
    try:
        return get_time(hook, "start")
    except (KeyError, ValueError):
        return None


def reorder_hook_first(sequence: list, hook_t) -> list:
    """Move the clip whose start matches hook_t to position 0."""
    if hook_t is None:
        log.warning("No hook defined — keeping sequence order as-is.")
        return sequence

    for i, clip in enumerate(sequence):
        try:
            s = get_time(clip, "start")
        except KeyError:
            continue
        if abs(s - hook_t) < 0.5:          # 500 ms tolerance
            if i == 0:
                log.info("Hook clip is already first.")
            else:
                log.info(f"Promoting clip at index {i} (start={s}s) to position 0 as hook.")
                sequence = [sequence[i]] + [c for j, c in enumerate(sequence) if j != i]
            return sequence

    log.warning(f"Hook start {hook_t}s not matched in sequence — keeping order as-is.")
    return sequence


# ---------------------------------------------------------------------------
# FFmpeg operations (all re-encode — no stream copy)
# ---------------------------------------------------------------------------

def cut_clip(ffmpeg: str, src: str, start: float, end: float, out: Path) -> None:
    duration = end - start
    if duration <= 0:
        raise ValueError(f"Clip end ({end:.3f}s) must be after start ({start:.3f}s).")

    cmd = [
        ffmpeg, "-y",
        "-ss", f"{start:.3f}",
        "-i", src,
        "-t", f"{duration:.3f}",
        "-c:v", VIDEO_CODEC, "-crf", CRF, "-preset", PRESET,
        "-c:a", AUDIO_CODEC, "-b:a", AUDIO_BITRATE, "-ar", SAMPLE_RATE,
        "-avoid_negative_ts", "make_zero",
        str(out),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        log.error("ffmpeg cut error:\n%s", result.stderr[-3000:])
        raise RuntimeError(f"Failed to cut clip → {out.name}")


def concat_clips(ffmpeg: str, clips: list, out: Path) -> None:
    list_file = out.parent / "_concat.txt"
    list_file.write_text(
        "".join(f"file '{p.as_posix()}'\n" for p in clips),
        encoding="utf-8",
    )
    cmd = [
        ffmpeg, "-y",
        "-f", "concat", "-safe", "0", "-i", str(list_file),
        "-c:v", VIDEO_CODEC, "-crf", CRF, "-preset", PRESET,
        "-c:a", AUDIO_CODEC, "-b:a", AUDIO_BITRATE, "-ar", SAMPLE_RATE,
        str(out),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        log.error("ffmpeg concat error:\n%s", result.stderr[-3000:])
        raise RuntimeError("Failed to concatenate clips.")


def render_scaled(ffmpeg: str, src: Path, vf: str, out: Path) -> None:
    cmd = [
        ffmpeg, "-y",
        "-i", str(src),
        "-vf", vf,
        "-c:v", VIDEO_CODEC, "-crf", CRF, "-preset", PRESET,
        "-c:a", AUDIO_CODEC, "-b:a", AUDIO_BITRATE, "-ar", SAMPLE_RATE,
        "-movflags", "+faststart",
        str(out),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        log.error("ffmpeg render error:\n%s", result.stderr[-3000:])
        raise RuntimeError(f"Failed to render {out.name}")


# ---------------------------------------------------------------------------
# Per-format pipeline
# ---------------------------------------------------------------------------

def process(ffmpeg: str, src: str, sequence: list, label: str,
            vf: str, out: Path, tmp: Path) -> None:
    """Cut → concat → scale for one output format."""
    log.info("-" * 60)
    log.info(f"Processing '{label}' — {len(sequence)} clip(s):")

    clip_files = []
    for i, clip in enumerate(sequence):
        try:
            start = get_time(clip, "start")
            end   = get_time(clip, "end")
        except (KeyError, ValueError) as exc:
            log.error(f"Clip {i} timestamp error: {exc}")
            sys.exit(1)

        clip_file = tmp / f"{label}_{i:03d}.mp4"
        log.info(f"  [{i:02d}] {start:.3f}s -> {end:.3f}s  ({end - start:.3f}s)")

        try:
            cut_clip(ffmpeg, src, start, end, clip_file)
        except (ValueError, RuntimeError) as exc:
            log.error(str(exc))
            sys.exit(2)

        clip_files.append(clip_file)

    log.info(f"  Assembling {len(clip_files)} clip(s)…")
    master = tmp / f"{label}_master.mp4"
    try:
        concat_clips(ffmpeg, clip_files, master)
    except RuntimeError as exc:
        log.error(str(exc))
        sys.exit(2)

    log.info(f"  Scaling -> {out.name}...")
    try:
        render_scaled(ffmpeg, master, vf, out)
    except RuntimeError as exc:
        log.error(str(exc))
        sys.exit(2)

    log.info(f"  Saved : {out.resolve()}  ({out.stat().st_size / (1024 * 1024):.1f} MB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 3:
        print(f"Usage: python {Path(__file__).name} <video_file> <gemini_output.json>")
        sys.exit(1)

    video_path = Path(sys.argv[1])
    json_path  = Path(sys.argv[2])

    if not video_path.exists():
        log.error(f"Video not found: {video_path}")
        sys.exit(1)
    if not json_path.exists():
        log.error(f"JSON not found: {json_path}")
        sys.exit(1)

    ffmpeg = check_ffmpeg()

    # --- Load JSON ---
    log.info("=" * 60)
    log.info("Loading gemini_output.json…")
    try:
        data = load_json(json_path)
        reel_seq, yt_seq = extract_sequences(data)
    except ValueError as exc:
        log.error(str(exc))
        sys.exit(1)

    hook_t = hook_start(data)
    log.info(f"  Hook start time : {hook_t}s" if hook_t is not None else "  Hook           : not defined")
    log.info(f"  Reel clips      : {len(reel_seq)}")
    log.info(f"  YouTube clips   : {len(yt_seq)}")

    reel_seq = reorder_hook_first(reel_seq, hook_t)
    yt_seq   = reorder_hook_first(yt_seq,   hook_t)

    out_dir  = Path.cwd()
    reel_out = out_dir / "reel_cut.mp4"
    yt_out   = out_dir / "youtube_cut.mp4"

    reel_vf = (
        f"scale={REEL_W}:{REEL_H}:force_original_aspect_ratio=increase,"
        f"crop={REEL_W}:{REEL_H}"
    )
    yt_vf = (
        f"scale={YT_W}:{YT_H}:force_original_aspect_ratio=decrease,"
        f"pad={YT_W}:{YT_H}:(ow-iw)/2:(oh-ih)/2:black"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        log.info("=" * 60)
        log.info("STAGE 1 — reel_cut.mp4  (90-second reel, 1080×1920, 9:16)")
        process(ffmpeg, str(video_path), reel_seq, "reel", reel_vf, reel_out, tmp)

        log.info("=" * 60)
        log.info("STAGE 2 — youtube_cut.mp4  (full cut, 1920×1080, 16:9)")
        process(ffmpeg, str(video_path), yt_seq, "yt", yt_vf, yt_out, tmp)

    log.info("=" * 60)
    log.info("Done.")
    log.info(f"  reel_cut.mp4    -> {reel_out.resolve()}")
    log.info(f"  youtube_cut.mp4 -> {yt_out.resolve()}")


if __name__ == "__main__":
    main()
