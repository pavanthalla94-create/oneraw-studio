import os
import sys
import time
import logging
import shutil
import subprocess
import json
from pathlib import Path

try:
    import requests
except ImportError:
    print("ERROR: 'requests' not installed. Run: pip install requests")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

REPLICATE_BASE  = "https://api.replicate.com/v1"
MODEL_OWNER     = "nightmareai"
MODEL_NAME      = "real-esrgan"
POLL_INTERVAL   = 15   # seconds between status checks

# 4K thresholds (UHD)
K4_WIDTH  = 3840
K4_HEIGHT = 2160


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_api_key() -> str:
    key = os.environ.get("REPLICATE_API_KEY", "").strip()
    if not key:
        log.error("REPLICATE_API_KEY environment variable is not set.")
        sys.exit(1)
    return key


def check_ffprobe() -> str:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        log.error("ffprobe not found in PATH. Install FFmpeg (includes ffprobe).")
        sys.exit(1)
    return ffprobe


def validate_input(path: str) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")
    if not p.is_file():
        raise ValueError(f"Not a file: {p}")
    return p


def auth_headers(api_key: str) -> dict:
    return {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }


# ---------------------------------------------------------------------------
# Stage 1: resolution check
# ---------------------------------------------------------------------------

def get_resolution(ffprobe: str, video_path: Path) -> tuple:
    """Return (width, height) using ffprobe JSON output."""
    cmd = [
        ffprobe, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "json",
        str(video_path),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed:\n{result.stderr}")

    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    if not streams:
        raise RuntimeError("ffprobe found no video streams in the file.")

    width  = streams[0].get("width",  0)
    height = streams[0].get("height", 0)
    return int(width), int(height)


def choose_scale(width: int, height: int) -> int:
    """Pick smallest scale factor that reaches 4K from the input resolution."""
    if height >= 1080 or width >= 1920:
        return 2   # 1080p x2 = 4K
    return 4       # 720p or lower: 4x gets closest to 4K


# ---------------------------------------------------------------------------
# Stage 2: upload to Replicate file storage
# ---------------------------------------------------------------------------

def upload_file(api_key: str, video_path: Path) -> str:
    """Upload video to Replicate's file API; return the hosted URL."""
    log.info(f"  Uploading {video_path.name}  ({video_path.stat().st_size / (1024*1024):.1f} MB)...")

    with open(video_path, "rb") as f:
        resp = requests.post(
            f"{REPLICATE_BASE}/files",
            headers=auth_headers(api_key),
            files={"content": (video_path.name, f, "video/mp4")},
            timeout=600,
        )

    if not resp.ok:
        log.error("File upload failed (%s): %s", resp.status_code, resp.text)
        raise RuntimeError("Replicate file upload failed.")

    body = resp.json()
    # Replicate returns { "urls": { "get": "https://..." }, ... }
    url = (
        body.get("urls", {}).get("get")
        or body.get("url")
        or body.get("download_url")
    )
    if not url:
        log.error("Unexpected upload response: %s", body)
        raise RuntimeError("No download URL returned from Replicate file upload.")

    log.info(f"  Upload complete. Hosted URL: {url}")
    return url


# ---------------------------------------------------------------------------
# Stage 3: create prediction
# ---------------------------------------------------------------------------

def create_prediction(api_key: str, video_url: str, scale: int) -> str:
    """Submit prediction; return prediction ID."""
    log.info(f"  Model  : {MODEL_OWNER}/{MODEL_NAME}  (scale={scale}x)")

    headers = {
        **auth_headers(api_key),
        "Content-Type": "application/json",
    }
    payload = {
        "input": {
            "video": video_url,
            "scale": scale,
        }
    }

    resp = requests.post(
        f"{REPLICATE_BASE}/models/{MODEL_OWNER}/{MODEL_NAME}/predictions",
        headers=headers,
        json=payload,
        timeout=30,
    )

    if not resp.ok:
        log.error("Prediction creation failed (%s): %s", resp.status_code, resp.text)
        raise RuntimeError("Failed to create Replicate prediction.")

    body = resp.json()
    pred_id = body.get("id")
    if not pred_id:
        log.error("Unexpected prediction response: %s", body)
        raise RuntimeError("No prediction ID in Replicate response.")

    log.info(f"  Prediction ID: {pred_id}")
    return pred_id


# ---------------------------------------------------------------------------
# Stage 4: poll until done
# ---------------------------------------------------------------------------

def poll_prediction(api_key: str, pred_id: str) -> dict:
    """Poll every POLL_INTERVAL seconds; return final prediction dict."""
    url = f"{REPLICATE_BASE}/predictions/{pred_id}"
    headers = auth_headers(api_key)
    elapsed = 0

    while True:
        resp = requests.get(url, headers=headers, timeout=30)
        if not resp.ok:
            log.error("Poll failed (%s): %s", resp.status_code, resp.text)
            raise RuntimeError("Error polling Replicate prediction.")

        data   = resp.json()
        status = data.get("status", "unknown")
        log.info(f"  [{elapsed:>4}s]  status: {status}")

        if status == "succeeded":
            log.info("  Prediction succeeded.")
            return data
        if status in ("failed", "canceled"):
            err = data.get("error", "no details")
            log.error("Prediction ended with status '%s': %s", status, err)
            raise RuntimeError(f"Replicate prediction {status}.")

        time.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL


# ---------------------------------------------------------------------------
# Stage 5: download output
# ---------------------------------------------------------------------------

def download_output(prediction: dict, out_path: Path) -> None:
    """Download the upscaled video from the prediction output URL."""
    output = prediction.get("output")
    if not output:
        raise RuntimeError("Prediction succeeded but 'output' field is empty.")

    # output may be a string URL or a list
    url = output if isinstance(output, str) else output[0]
    log.info(f"  Downloading from: {url}")

    with requests.get(url, stream=True, timeout=600) as resp:
        resp.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    log.info(f"  Saved: {out_path.resolve()}  ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: python {Path(__file__).name} <input_video>")
        sys.exit(1)

    try:
        video_path = validate_input(sys.argv[1])
    except (FileNotFoundError, ValueError) as exc:
        log.error(str(exc))
        sys.exit(1)

    api_key = get_api_key()
    ffprobe = check_ffprobe()
    out_path = Path.cwd() / "upscaled_video.mp4"

    # --- Stage 1: resolution check ---
    log.info("=" * 60)
    log.info("STAGE 1 -- Checking video resolution...")
    try:
        width, height = get_resolution(ffprobe, video_path)
    except RuntimeError as exc:
        log.error(str(exc))
        sys.exit(1)

    log.info(f"  Resolution: {width}x{height}")

    if width >= K4_WIDTH or height >= K4_HEIGHT:
        log.info(f"  Already 4K or higher ({width}x{height}). Nothing to do.")
        sys.exit(0)

    scale = choose_scale(width, height)
    log.info(f"  Not 4K. Will upscale {scale}x  ({width}x{height} -> {width*scale}x{height*scale})")

    # --- Stage 2: upload ---
    log.info("=" * 60)
    log.info("STAGE 2 -- Uploading video to Replicate...")
    try:
        video_url = upload_file(api_key, video_path)
    except RuntimeError as exc:
        log.error(str(exc))
        sys.exit(2)

    # --- Stage 3: create prediction ---
    log.info("=" * 60)
    log.info("STAGE 3 -- Creating upscale prediction...")
    try:
        pred_id = create_prediction(api_key, video_url, scale)
    except RuntimeError as exc:
        log.error(str(exc))
        sys.exit(2)

    # --- Stage 4: poll ---
    log.info("=" * 60)
    log.info(f"STAGE 4 -- Polling every {POLL_INTERVAL}s until complete...")
    try:
        prediction = poll_prediction(api_key, pred_id)
    except RuntimeError as exc:
        log.error(str(exc))
        sys.exit(2)

    # --- Stage 5: download ---
    log.info("=" * 60)
    log.info("STAGE 5 -- Downloading upscaled video...")
    try:
        download_output(prediction, out_path)
    except (RuntimeError, requests.RequestException) as exc:
        log.error("Download failed: %s", exc)
        sys.exit(2)

    log.info("=" * 60)
    log.info("Done.")
    log.info(f"  upscaled_video.mp4 -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
