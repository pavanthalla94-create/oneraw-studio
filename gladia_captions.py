import os
import sys
import time
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

try:
    import requests
except ImportError:
    print("ERROR: 'requests' is not installed. Run: pip install requests")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

GLADIA_BASE = "https://api.gladia.io/v2"
POLL_INTERVAL = 5  # seconds between status checks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_api_key() -> str:
    key = os.environ.get("GLADIA_API_KEY", "").strip()
    if not key:
        log.error("GLADIA_API_KEY environment variable is not set.")
        sys.exit(1)
    return key


def check_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        log.error("ffmpeg not found in PATH. Install it from https://ffmpeg.org/download.html")
        sys.exit(1)
    return ffmpeg


def validate_input(path: str) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")
    if not p.is_file():
        raise ValueError(f"Path is not a file: {p}")
    return p


def seconds_to_srt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# ---------------------------------------------------------------------------
# Stage 1 — extract audio
# ---------------------------------------------------------------------------

def extract_audio(video_path: Path, mp3_path: Path) -> None:
    log.info("=" * 60)
    log.info("STAGE 1: Extracting audio from video…")
    log.info(f"  Source : {video_path}")
    log.info(f"  Output : {mp3_path}")

    ffmpeg = check_ffmpeg()
    cmd = [
        ffmpeg, "-y",
        "-i", str(video_path),
        "-vn",                  # drop video stream
        "-acodec", "libmp3lame",
        "-b:a", "128k",
        "-ar", "44100",
        str(mp3_path),
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        log.error("ffmpeg error:\n%s", result.stderr)
        raise RuntimeError("Audio extraction failed.")

    size_mb = mp3_path.stat().st_size / (1024 * 1024)
    log.info(f"  Done   : {size_mb:.2f} MB extracted.")


# ---------------------------------------------------------------------------
# Stage 2 — upload to Gladia
# ---------------------------------------------------------------------------

def upload_audio(api_key: str, mp3_path: Path) -> str:
    log.info("=" * 60)
    log.info("STAGE 2: Uploading MP3 to Gladia…")

    headers = {"x-gladia-key": api_key}

    with open(mp3_path, "rb") as f:
        resp = requests.post(
            f"{GLADIA_BASE}/upload",
            headers=headers,
            files={"audio": (mp3_path.name, f, "audio/mpeg")},
            timeout=120,
        )

    if not resp.ok:
        log.error("Upload failed (%s): %s", resp.status_code, resp.text)
        raise RuntimeError("Gladia upload failed.")

    audio_url = resp.json().get("audio_url") or resp.json().get("url")
    if not audio_url:
        log.error("Unexpected upload response: %s", resp.json())
        raise RuntimeError("No audio_url returned by Gladia.")

    log.info(f"  Uploaded. audio_url: {audio_url}")
    return audio_url


# ---------------------------------------------------------------------------
# Stage 3 — start transcription
# ---------------------------------------------------------------------------

def start_transcription(api_key: str, audio_url: str) -> str:
    log.info("=" * 60)
    log.info("STAGE 3: Starting transcription (auto language detection)…")

    headers = {
        "x-gladia-key": api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "audio_url": audio_url,
        "detect_language": True,
        "diarization": False,
    }

    resp = requests.post(
        f"{GLADIA_BASE}/transcription",
        headers=headers,
        json=payload,
        timeout=30,
    )

    if not resp.ok:
        log.error("Transcription start failed (%s): %s", resp.status_code, resp.text)
        raise RuntimeError("Failed to start Gladia transcription.")

    job_id = resp.json().get("id") or resp.json().get("result_url", "").split("/")[-1]
    if not job_id:
        log.error("Unexpected response: %s", resp.json())
        raise RuntimeError("No job ID returned by Gladia.")

    log.info(f"  Job ID : {job_id}")
    return job_id


# ---------------------------------------------------------------------------
# Stage 4 — poll until done
# ---------------------------------------------------------------------------

def poll_transcription(api_key: str, job_id: str) -> dict:
    log.info("=" * 60)
    log.info(f"STAGE 4: Polling transcription (every {POLL_INTERVAL}s)…")

    headers = {"x-gladia-key": api_key}
    url = f"{GLADIA_BASE}/transcription/{job_id}"

    while True:
        resp = requests.get(url, headers=headers, timeout=30)

        if not resp.ok:
            log.error("Poll failed (%s): %s", resp.status_code, resp.text)
            raise RuntimeError("Error while polling Gladia.")

        data = resp.json()
        status = data.get("status", "unknown")
        log.info(f"  Status : {status}")

        if status == "done":
            log.info("  Transcription complete.")
            return data
        if status in ("error", "failed"):
            log.error("Gladia job failed: %s", data)
            raise RuntimeError(f"Gladia transcription ended with status: {status}")

        time.sleep(POLL_INTERVAL)


# ---------------------------------------------------------------------------
# Stage 5 — save SRT + TXT
# ---------------------------------------------------------------------------

def save_outputs(result: dict, srt_path: Path, txt_path: Path) -> None:
    log.info("=" * 60)
    log.info("STAGE 5: Saving outputs…")

    # Navigate Gladia v2 response structure
    prediction = (
        result.get("result", {})
              .get("transcription", {})
              .get("utterances")
        or result.get("result", {})
                 .get("transcription", {})
                 .get("full_transcript")
        or []
    )

    utterances = result.get("result", {}).get("transcription", {}).get("utterances", [])

    # --- SRT ---
    srt_lines = []
    for i, utt in enumerate(utterances, start=1):
        start = seconds_to_srt_time(utt.get("start", 0.0))
        end = seconds_to_srt_time(utt.get("end", 0.0))
        text = utt.get("text", "").strip()
        srt_lines.append(f"{i}\n{start} --> {end}\n{text}\n")

    srt_content = "\n".join(srt_lines)
    srt_path.write_text(srt_content, encoding="utf-8")
    log.info(f"  SRT saved     : {srt_path.resolve()}  ({len(utterances)} entries)")

    # --- Plain transcript ---
    full_text = (
        result.get("result", {})
              .get("transcription", {})
              .get("full_transcript", "")
        or " ".join(u.get("text", "") for u in utterances)
    ).strip()

    txt_path.write_text(full_text, encoding="utf-8")
    log.info(f"  TXT saved     : {txt_path.resolve()}")

    detected_lang = (
        result.get("result", {})
              .get("transcription", {})
              .get("languages", ["unknown"])
    )
    log.info(f"  Detected lang : {detected_lang}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: python {Path(__file__).name} <input_video>")
        sys.exit(1)

    video_path = validate_input(sys.argv[1])
    api_key = get_api_key()

    out_dir = Path.cwd()
    srt_path = out_dir / "captions.srt"
    txt_path = out_dir / "transcript.txt"

    with tempfile.TemporaryDirectory() as tmpdir:
        mp3_path = Path(tmpdir) / (video_path.stem + ".mp3")

        try:
            extract_audio(video_path, mp3_path)
            audio_url = upload_audio(api_key, mp3_path)
            job_id = start_transcription(api_key, audio_url)
            result = poll_transcription(api_key, job_id)
            save_outputs(result, srt_path, txt_path)
        except (FileNotFoundError, ValueError) as exc:
            log.error(str(exc))
            sys.exit(1)
        except RuntimeError as exc:
            log.error(str(exc))
            sys.exit(2)
        except requests.RequestException as exc:
            log.error("Network error: %s", exc)
            sys.exit(3)

    log.info("=" * 60)
    log.info("Done.")
    log.info(f"  captions.srt  -> {srt_path.resolve()}")
    log.info(f"  transcript.txt -> {txt_path.resolve()}")


if __name__ == "__main__":
    main()
