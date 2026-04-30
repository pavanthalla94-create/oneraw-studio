#!/usr/bin/env python3
"""
gemini_analyze.py - Analyze video using Google Gemini File API and Gemini 2.0 Flash.

Usage:
    python gemini_analyze.py <video_path> [--output <output.json>]

Requires:
    pip install google-genai
    GEMINI_API_KEY environment variable set
    ffprobe (ffmpeg) installed for timestamp validation
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

# ── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("gemini_analyze.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
DEFAULT_OUTPUT = "gemini_output.json"
MODEL = "gemini-2.5-flash"
UPLOAD_TIMEOUT_SECONDS = 600
UPLOAD_POLL_INTERVAL = 5

PROMPT = (
    "You are a professional video editor with 15 years experience. "
    "Watch this video carefully. Return a JSON object with these fields: "
    "best_moments as array of start_time end_time reason, "
    "hook as the single best 3-second moment for opening, "
    "avoid as array of start_time end_time reason, "
    "total_duration as video length in seconds, "
    "suggested_sequence as ordered array of clip timestamps for a 90 second reel "
    "and 8 minute youtube video"
)

MIME_MAP = {
    ".mp4": "video/mp4",
    ".m4v": "video/mp4",
    ".mov": "video/quicktime",
    ".avi": "video/x-msvideo",
    ".mkv": "video/x-matroska",
    ".webm": "video/webm",
    ".flv": "video/x-flv",
    ".wmv": "video/x-ms-wmv",
    ".3gp": "video/3gpp",
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def stage(label: str, message: str) -> None:
    """Print and log a clearly labelled stage update."""
    line = f"[{label}] {message}"
    print(line)
    logger.info(line)


def get_video_duration(video_path: str) -> float | None:
    """Return video duration in seconds via ffprobe, or None if unavailable."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                video_path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            logger.warning("ffprobe non-zero exit: %s", result.stderr.strip())
            return None
        probe = json.loads(result.stdout)
        duration = float(probe["format"]["duration"])
        logger.info("ffprobe duration: %.2f seconds", duration)
        return duration
    except FileNotFoundError:
        logger.warning("ffprobe not found — install ffmpeg for timestamp validation")
        return None
    except (subprocess.TimeoutExpired, json.JSONDecodeError, KeyError, ValueError) as exc:
        logger.warning("Could not parse video duration: %s", exc)
        return None


def _ts_ok(value, max_dur: float) -> bool:
    """Return True if value is a number in [0, max_dur]."""
    try:
        return 0.0 <= float(value) <= max_dur
    except (TypeError, ValueError):
        return False


def _moment_ok(moment: object, max_dur: float) -> bool:
    """Return True if moment is a dict with valid start_time and end_time."""
    return (
        isinstance(moment, dict)
        and _ts_ok(moment.get("start_time"), max_dur)
        and _ts_ok(moment.get("end_time"), max_dur)
    )


def validate_timestamps(data: dict, max_duration: float | None) -> dict:
    """
    Remove any entries whose timestamps exceed the actual video duration.
    Handles best_moments, avoid, hook, and suggested_sequence (flat list or
    reel/youtube dict).
    """
    if max_duration is None:
        logger.warning("No video duration — skipping timestamp validation")
        return data

    def _filter_list(key: str) -> None:
        if key not in data or not isinstance(data[key], list):
            return
        before = len(data[key])
        data[key] = [m for m in data[key] if _moment_ok(m, max_duration)]
        removed = before - len(data[key])
        if removed:
            stage("VALIDATING", f"Removed {removed} '{key}' entries with out-of-range timestamps")

    _filter_list("best_moments")
    _filter_list("avoid")

    # Hook is a single object, not a list
    if "hook" in data and isinstance(data["hook"], dict):
        if not _moment_ok(data["hook"], max_duration):
            stage("VALIDATING", "Hook timestamps exceed video duration — clearing hook")
            data["hook"] = None

    # suggested_sequence can be a flat list of dicts/numbers or a dict with
    # sub-lists (e.g. {"reel": [...], "youtube": [...]})
    if "suggested_sequence" in data:
        seq = data["suggested_sequence"]

        if isinstance(seq, dict):
            for sub_key, sub_list in list(seq.items()):
                if not isinstance(sub_list, list):
                    continue
                before = len(sub_list)
                seq[sub_key] = [m for m in sub_list if _moment_ok(m, max_duration)]
                removed = before - len(seq[sub_key])
                if removed:
                    stage("VALIDATING", f"Removed {removed} suggested_sequence['{sub_key}'] entries")

        elif isinstance(seq, list):
            before = len(seq)
            valid = []
            for item in seq:
                if isinstance(item, dict):
                    if _moment_ok(item, max_duration):
                        valid.append(item)
                elif isinstance(item, (int, float)):
                    if _ts_ok(item, max_duration):
                        valid.append(item)
                else:
                    valid.append(item)  # keep non-timestamp items untouched
            removed = before - len(valid)
            if removed:
                stage("VALIDATING", f"Removed {removed} suggested_sequence entries with out-of-range timestamps")
            data["suggested_sequence"] = valid

    return data


def parse_json_response(raw: str) -> dict:
    """Parse JSON from Gemini's response, stripping markdown fences if present."""
    text = raw.strip()

    # Strip ```json … ``` or ``` … ``` fences
    if text.startswith("```"):
        lines = text.splitlines()
        inner: list[str] = []
        in_block = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("```") and not in_block:
                in_block = True
                continue
            if stripped == "```" and in_block:
                break
            if in_block:
                inner.append(line)
        text = "\n".join(inner)

    return json.loads(text)


# ── Core operations ──────────────────────────────────────────────────────────

def upload_video(client, video_path: str):
    """
    Upload video to Gemini File API and poll until the file is ACTIVE.
    Returns the ready file object.
    """
    size_mb = Path(video_path).stat().st_size / (1024 * 1024)
    stage("UPLOADING", f"Uploading '{Path(video_path).name}' ({size_mb:.1f} MB) to Gemini File API…")

    uploaded = client.files.upload(file=video_path)
    logger.info("Upload initiated — name: %s  uri: %s", uploaded.name, uploaded.uri)

    stage("UPLOADING", f"File received by API. Waiting for processing…  (URI: {uploaded.uri})")

    elapsed = 0
    state_name = lambda f: f.state.name if hasattr(f.state, "name") else str(f.state)

    while state_name(uploaded) == "PROCESSING":
        if elapsed >= UPLOAD_TIMEOUT_SECONDS:
            raise TimeoutError(
                f"File still PROCESSING after {UPLOAD_TIMEOUT_SECONDS}s — "
                "try again or check the Gemini console"
            )
        time.sleep(UPLOAD_POLL_INTERVAL)
        elapsed += UPLOAD_POLL_INTERVAL
        uploaded = client.files.get(name=uploaded.name)
        logger.debug("State: %s  elapsed: %ds", state_name(uploaded), elapsed)

    if state_name(uploaded) == "FAILED":
        raise RuntimeError(f"Gemini file processing failed for '{uploaded.name}'")

    stage("UPLOADING", f"File ready. State: {state_name(uploaded)}")
    return uploaded


def analyze_video(client, uploaded_file) -> str:
    """Send uploaded file to Gemini 2.0 Flash with the analysis prompt."""
    from google.genai import types  # imported here so failure is caught later

    mime = (
        uploaded_file.mime_type
        or MIME_MAP.get(Path(uploaded_file.display_name or "").suffix.lower(), "video/mp4")
        if hasattr(uploaded_file, "display_name")
        else "video/mp4"
    )

    stage("ANALYZING", f"Sending video to {MODEL} for analysis…")
    logger.info("File URI: %s  MIME: %s", uploaded_file.uri, mime)

    response = client.models.generate_content(
        model=MODEL,
        contents=[
            types.Content(parts=[
                types.Part(
                    file_data=types.FileData(
                        file_uri=uploaded_file.uri,
                        mime_type=mime,
                    )
                ),
                types.Part(text=PROMPT),
            ])
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
        ),
    )

    text = response.text
    logger.info("Response received: %d characters", len(text))
    stage("ANALYZING", "Analysis complete.")
    return text


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze a video with Google Gemini 2.0 Flash via the File API"
    )
    parser.add_argument("video_path", help="Path to the video file to analyze")
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output JSON file path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    # ── Validate API key ─────────────────────────────────────────────────────
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY environment variable is not set")
        print("[ERROR] Set the GEMINI_API_KEY environment variable before running.")
        sys.exit(1)

    # ── Validate input file ──────────────────────────────────────────────────
    video_path = Path(args.video_path).resolve()
    if not video_path.exists():
        logger.error("File not found: %s", video_path)
        print(f"[ERROR] File not found: {video_path}")
        sys.exit(1)
    if not video_path.is_file():
        logger.error("Path is not a file: %s", video_path)
        print(f"[ERROR] Not a file: {video_path}")
        sys.exit(1)

    print(f"\n{'='*62}")
    print("  Gemini Video Analyzer")
    print(f"{'='*62}")
    print(f"  Video  : {video_path}")
    print(f"  Output : {args.output}")
    print(f"  Model  : {MODEL}")
    print(f"{'='*62}")

    # ── Get video duration for timestamp validation ───────────────────────────
    stage("METADATA", "Reading video metadata with ffprobe…")
    video_duration = get_video_duration(str(video_path))
    if video_duration is not None:
        stage("METADATA", f"Duration: {video_duration:.2f}s ({video_duration / 60:.1f} min)")
    else:
        stage(
            "METADATA",
            "Duration unavailable (ffprobe missing or failed) — timestamp validation will be skipped",
        )

    # ── Import google-genai ───────────────────────────────────────────────────
    try:
        from google import genai  # noqa: F401
    except ImportError:
        print("[ERROR] google-genai package not installed.")
        print("        Run: pip install google-genai")
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    raw_response: str | None = None
    output_path = Path(args.output)

    try:
        # 1. Upload
        uploaded_file = upload_video(client, str(video_path))

        # 2. Analyze
        raw_response = analyze_video(client, uploaded_file)

        # 3. Parse JSON
        stage("PARSING", "Parsing JSON response…")
        try:
            analysis = parse_json_response(raw_response)
        except json.JSONDecodeError as exc:
            logger.error("JSON parse error: %s", exc)
            raw_path = output_path.with_suffix(".raw.txt")
            raw_path.write_text(raw_response, encoding="utf-8")
            stage("PARSING", f"Failed to parse JSON. Raw response saved to {raw_path}")
            sys.exit(1)
        stage("PARSING", "JSON parsed successfully.")

        # 4. Validate timestamps
        stage("VALIDATING", f"Validating timestamps (max duration: {video_duration}s)…")
        analysis = validate_timestamps(analysis, video_duration)
        stage("VALIDATING", "Timestamp validation complete.")

        # 5. Attach metadata
        analysis["_metadata"] = {
            "video_file": str(video_path),
            "file_uri": uploaded_file.uri,
            "analyzed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model": MODEL,
            "actual_duration_seconds": video_duration,
        }

        # 6. Save
        stage("SAVING", f"Writing analysis to {output_path}…")
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(analysis, fh, indent=2, ensure_ascii=False)
        size_kb = output_path.stat().st_size / 1024
        stage("SAVING", f"Saved successfully — {output_path.resolve()} ({size_kb:.1f} KB)")

        # ── Summary ──────────────────────────────────────────────────────────
        best = analysis.get("best_moments", [])
        avoid = analysis.get("avoid", [])
        hook = analysis.get("hook")
        total = analysis.get("total_duration", video_duration)

        print(f"\n{'='*62}")
        print("  Analysis Complete")
        print(f"{'='*62}")
        print(f"  Best moments found : {len(best) if isinstance(best, list) else 'N/A'}")
        print(f"  Sections to avoid  : {len(avoid) if isinstance(avoid, list) else 'N/A'}")
        if isinstance(hook, dict):
            print(f"  Hook               : {hook.get('start_time')}s – {hook.get('end_time')}s")
        if total is not None:
            print(f"  Total duration     : {total}s")
        print(f"  Output file        : {output_path.resolve()}")
        print(f"{'='*62}\n")
        logger.info("Done.")

    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Cancelled by user")
        logger.info("Cancelled by user")
        sys.exit(130)

    except TimeoutError as exc:
        logger.error("Timeout: %s", exc)
        print(f"\n[ERROR] Timeout: {exc}")
        sys.exit(1)

    except RuntimeError as exc:
        logger.error("Runtime error: %s", exc)
        print(f"\n[ERROR] {exc}")
        sys.exit(1)

    except Exception as exc:  # noqa: BLE001
        logger.error("Unexpected error: %s", exc, exc_info=True)
        print(f"\n[ERROR] Unexpected error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
