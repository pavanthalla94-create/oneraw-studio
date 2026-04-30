import sys
import os
import re
import time
import json
import base64
import logging
import shutil
import subprocess
import argparse
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

EDIT_BASE     = "https://api.shotstack.io/edit/stage"
INGEST_BASE   = "https://api.shotstack.io/ingest/stage"
POLL_INTERVAL = 10   # seconds between status polls


# ---------------------------------------------------------------------------
# Environment / setup
# ---------------------------------------------------------------------------

def get_api_key() -> str:
    key = os.environ.get("SHOTSTACK_API_KEY", "").strip()
    if not key:
        log.error("SHOTSTACK_API_KEY environment variable is not set.")
        sys.exit(1)
    return key


def check_ffprobe() -> str:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        log.error("ffprobe not found in PATH. Install FFmpeg (includes ffprobe).")
        sys.exit(1)
    return ffprobe


def edit_headers(api_key: str) -> dict:
    return {"x-api-key": api_key, "Content-Type": "application/json"}


def bare_headers(api_key: str) -> dict:
    return {"x-api-key": api_key}


# ---------------------------------------------------------------------------
# Stage 1: get video duration via ffprobe
# ---------------------------------------------------------------------------

def get_duration(ffprobe: str, video_path: Path) -> float:
    cmd = [
        ffprobe, "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        str(video_path),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed:\n{result.stderr}")
    data = json.loads(result.stdout)
    duration = data.get("format", {}).get("duration")
    if duration is None:
        raise RuntimeError("ffprobe returned no duration for this file.")
    return float(duration)


# ---------------------------------------------------------------------------
# Stage 2: upload video
#   Shotstack Ingest API is URL-only; Serve API caps at 10 MB.
#   Strategy for large local files:
#     A) Try multiple free temp hosts in order until one succeeds, OR
#        use SHOTSTACK_VIDEO_URL env var to skip upload entirely.
#     B) Ingest that URL via Shotstack → get CDN-hosted URL.
#     C) Poll ingest until status == "ready".
#   CDN URL at data.attributes.source is used in the Edit timeline.
# ---------------------------------------------------------------------------

def _try_0x0st(video_path: Path) -> str:
    with open(video_path, "rb") as f:
        resp = requests.post(
            "https://0x0.st/",
            files={"file": (video_path.name, f, "video/mp4")},
            timeout=600,
        )
    if not resp.ok:
        raise RuntimeError(f"0x0.st responded {resp.status_code}")
    return resp.text.strip()


def _try_fileio(video_path: Path) -> str:
    with open(video_path, "rb") as f:
        resp = requests.post(
            "https://file.io/?expires=1d",
            files={"file": (video_path.name, f, "video/mp4")},
            timeout=600,
        )
    if not resp.ok:
        raise RuntimeError(f"file.io responded {resp.status_code}")
    data = resp.json()
    if not data.get("success"):
        raise RuntimeError(f"file.io error: {data}")
    return data["link"]


def _try_tmpfiles(video_path: Path) -> str:
    with open(video_path, "rb") as f:
        resp = requests.post(
            "https://tmpfiles.org/api/v1/upload",
            files={"file": (video_path.name, f, "video/mp4")},
            timeout=600,
        )
    if not resp.ok:
        raise RuntimeError(f"tmpfiles.org responded {resp.status_code}")
    # returns {"status":"success","data":{"url":"https://tmpfiles.org/..."}}
    page_url = resp.json()["data"]["url"]
    # convert page URL to direct download URL
    return page_url.replace("tmpfiles.org/", "tmpfiles.org/dl/")


TEMP_HOSTS = [
    ("0x0.st",       _try_0x0st),
    ("file.io",      _try_fileio),
    ("tmpfiles.org", _try_tmpfiles),
]


def _upload_to_temp_host(video_path: Path) -> str:
    for name, fn in TEMP_HOSTS:
        try:
            log.info(f"  Trying {name}...")
            url = fn(video_path)
            log.info(f"  Temporary URL ({name}): {url}")
            return url
        except Exception as exc:
            log.warning(f"  {name} failed: {exc}")

    raise RuntimeError(
        "All temp-hosting providers failed.\n"
        "  Set SHOTSTACK_VIDEO_URL=<public_url> to provide a pre-hosted URL and re-run."
    )


def _ingest_url(api_key: str, source_url: str) -> str:
    """Ingest a public URL into Shotstack and return its CDN URL."""
    log.info("  Ingesting into Shotstack...")
    resp = requests.post(
        f"{INGEST_BASE}/sources",
        headers={**bare_headers(api_key), "Content-Type": "application/json"},
        json={"url": source_url},
        timeout=30,
    )
    if not resp.ok:
        log.error("Shotstack ingest failed (%s): %s", resp.status_code, resp.text)
        raise RuntimeError("Shotstack ingest request failed.")

    source_id = resp.json()["data"]["id"]
    log.info(f"  Ingest source ID : {source_id}")

    poll_url = f"{INGEST_BASE}/sources/{source_id}"
    elapsed  = 0
    while True:
        resp = requests.get(poll_url, headers=bare_headers(api_key), timeout=30)
        if not resp.ok:
            log.error("Ingest poll failed (%s): %s", resp.status_code, resp.text)
            raise RuntimeError("Error polling Shotstack ingest.")

        attrs  = resp.json()["data"]["attributes"]
        status = attrs.get("status", "unknown")
        log.info(f"  Ingest [{elapsed:>3}s] : {status}")

        if status == "ready":
            cdn_url = attrs.get("source") or attrs.get("url")
            log.info(f"  CDN URL : {cdn_url}")
            return cdn_url
        if status in ("failed", "error", "deleted"):
            raise RuntimeError(f"Shotstack ingest ended with status: {status}")

        time.sleep(5)
        elapsed += 5


def upload_video(api_key: str, video_path: Path) -> str:
    size_mb = video_path.stat().st_size / (1024 * 1024)
    log.info(f"  File : {video_path.name}  ({size_mb:.1f} MB)")

    # Allow bypassing upload if video is already hosted
    pre_url = os.environ.get("SHOTSTACK_VIDEO_URL", "").strip()
    if pre_url:
        log.info(f"  Using pre-hosted URL from SHOTSTACK_VIDEO_URL: {pre_url}")
        return _ingest_url(api_key, pre_url)

    temp_url = _upload_to_temp_host(video_path)
    return _ingest_url(api_key, temp_url)


# ---------------------------------------------------------------------------
# Stage 3: encode local logo to base64 data URI (no upload needed)
# ---------------------------------------------------------------------------

def encode_logo(logo_path: Path) -> str:
    data = logo_path.read_bytes()
    b64  = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{b64}"


# ---------------------------------------------------------------------------
# Stage 4: parse captions.srt
# ---------------------------------------------------------------------------

def parse_srt(srt_path: Path) -> list:
    content = srt_path.read_text(encoding="utf-8-sig", errors="replace")
    # match each subtitle block: index / timestamps / text
    pattern = re.compile(
        r"\d+\r?\n"
        r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*"
        r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\r?\n"
        r"([\s\S]+?)(?=\r?\n\s*\r?\n|\Z)",
    )
    entries = []
    for m in pattern.finditer(content):
        h1, m1, s1, ms1, h2, m2, s2, ms2, text = m.groups()
        start  = int(h1) * 3600 + int(m1) * 60 + int(s1) + int(ms1) / 1000
        end    = int(h2) * 3600 + int(m2) * 60 + int(s2) + int(ms2) / 1000
        text   = re.sub(r"\r?\n", " ", text.strip())
        entries.append({"start": round(start, 3), "end": round(end, 3), "text": text})
    return entries


# ---------------------------------------------------------------------------
# HTML overlay builders
# ---------------------------------------------------------------------------

GRAIN_HTML = (
    '<html><head><style>'
    '*{margin:0;padding:0}'
    'body{width:100%;height:100%;background:transparent;overflow:hidden}'
    'svg{display:block;width:100%;height:100%}'
    '</style></head><body>'
    '<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="100%">'
    '<filter id="g" x="0" y="0" width="100%" height="100%">'
    '<feTurbulence type="fractalNoise" baseFrequency="0.80" numOctaves="4" stitchTiles="stitch"/>'
    '<feColorMatrix type="saturate" values="0"/>'
    '</filter>'
    '<rect width="100%" height="100%" filter="url(#g)"/>'
    '</svg></body></html>'
)

VIGNETTE_HTML = (
    '<html><head><style>'
    '*{margin:0;padding:0}'
    'body{'
    'width:100%;height:100%;'
    'background:radial-gradient('
    'ellipse at 50% 50%,'
    'transparent 52%,'
    'rgba(0,0,0,0.78) 100%)'
    '}'
    '</style></head><body></body></html>'
)


# ---------------------------------------------------------------------------
# Stage 5: build Shotstack timeline
# Track order: first in array = TOP layer, last = BOTTOM layer.
# ---------------------------------------------------------------------------

def build_timeline(
    video_url: str,
    duration: float,
    logo_uri: str | None,
    captions: list,
    width: int,
    height: int,
) -> dict:
    tracks = []

    # --- TOP: film grain (subtle monochrome noise) ---
    tracks.append({
        "clips": [{
            "asset": {
                "type": "html",
                "html": GRAIN_HTML,
                "width": width,
                "height": height,
                "background": "transparent",
            },
            "start": 0,
            "length": duration,
            "opacity": 0.09,
        }]
    })

    # --- Captions track ---
    if captions:
        caption_clips = []
        for c in captions:
            clip_len = max(round(c["end"] - c["start"], 3), 0.1)
            caption_clips.append({
                "asset": {
                    "type": "title",
                    "text": c["text"],
                    "style": "minimal",
                    "color": "#ffffff",
                    "size": "small",
                    "background": "transparent",
                },
                "start": c["start"],
                "length": clip_len,
                "position": "bottom",
                "offset": {"x": 0, "y": 0.12},
            })
        tracks.append({"clips": caption_clips})

    # --- Logo watermark (bottom-right) ---
    if logo_uri:
        tracks.append({
            "clips": [{
                "asset": {
                    "type": "image",
                    "src": logo_uri,
                },
                "start": 0,
                "length": duration,
                "position": "bottomRight",
                "offset": {"x": -0.04, "y": 0.05},
                "scale": 0.14,
                "opacity": 0.85,
            }]
        })

    # --- Vignette ---
    tracks.append({
        "clips": [{
            "asset": {
                "type": "html",
                "html": VIGNETTE_HTML,
                "width": width,
                "height": height,
                "background": "transparent",
            },
            "start": 0,
            "length": duration,
            "opacity": 1.0,
        }]
    })

    # --- BOTTOM: main video ---
    tracks.append({
        "clips": [{
            "asset": {
                "type": "video",
                "src": video_url,
                "volume": 1.0,
            },
            "start": 0,
            "length": duration,
        }]
    })

    return {
        "timeline": {
            "background": "#000000",
            "tracks": tracks,
        }
    }


# ---------------------------------------------------------------------------
# Stage 6: submit render jobs
# ---------------------------------------------------------------------------

def submit_render(api_key: str, timeline: dict, aspect_ratio: str, label: str) -> str:
    payload = {
        **timeline,
        "output": {
            "format": "mp4",
            "resolution": "1080",
            "aspectRatio": aspect_ratio,
        },
    }
    resp = requests.post(
        f"{EDIT_BASE}/render",
        headers=edit_headers(api_key),
        json=payload,
        timeout=30,
    )
    if not resp.ok:
        log.error("%s render submit failed (%s):\n%s", label, resp.status_code, resp.text)
        raise RuntimeError(f"Failed to submit {label} render.")

    render_id = resp.json()["response"]["id"]
    log.info(f"  {label} render ID: {render_id}")
    return render_id


# ---------------------------------------------------------------------------
# Stage 7: poll both render jobs simultaneously
# ---------------------------------------------------------------------------

def poll_both(api_key: str, yt_id: str, reel_id: str) -> tuple:
    jobs = {
        "YouTube": {"id": yt_id,   "done": False, "url": None},
        "Reels":   {"id": reel_id, "done": False, "url": None},
    }
    elapsed = 0

    while not all(j["done"] for j in jobs.values()):
        for label, job in jobs.items():
            if job["done"]:
                continue
            resp = requests.get(
                f"{EDIT_BASE}/render/{job['id']}",
                headers={"x-api-key": api_key},
                timeout=30,
            )
            if not resp.ok:
                log.error("%s poll failed (%s): %s", label, resp.status_code, resp.text)
                raise RuntimeError(f"Error polling {label} render.")

            data   = resp.json()["response"]
            status = data["status"]
            log.info(f"  {label:<8} [{elapsed:>4}s]: {status}")

            if status == "done":
                job["url"]  = data["url"]
                job["done"] = True
            elif status in ("failed", "error"):
                err = data.get("error", "no details")
                raise RuntimeError(f"{label} render failed: {err}")

        if not all(j["done"] for j in jobs.values()):
            time.sleep(POLL_INTERVAL)
            elapsed += POLL_INTERVAL

    return jobs["YouTube"]["url"], jobs["Reels"]["url"]


# ---------------------------------------------------------------------------
# Stage 8: download files
# ---------------------------------------------------------------------------

def download_file(url: str, out_path: Path) -> None:
    with requests.get(url, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    log.info(f"  {out_path.name}  ({size_mb:.1f} MB)  -> {out_path.resolve()}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render video with Shotstack (captions + grain + vignette + logo)"
    )
    parser.add_argument("video", help="Path to input video file")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        log.error(f"Video not found: {video_path}")
        sys.exit(1)

    api_key  = get_api_key()
    ffprobe  = check_ffprobe()
    cwd      = Path.cwd()

    logo_path     = cwd / "logo.png"
    captions_path = cwd / "captions.srt"
    yt_out        = cwd / "final_youtube.mp4"
    reel_out      = cwd / "final_reel.mp4"

    # --- Stage 1: duration ---
    log.info("=" * 60)
    log.info("STAGE 1 -- Detecting video duration...")
    try:
        duration = get_duration(ffprobe, video_path)
    except RuntimeError as exc:
        log.error(str(exc))
        sys.exit(1)
    log.info(f"  Duration: {duration:.3f}s")

    # --- Stage 2: upload video ---
    log.info("=" * 60)
    log.info("STAGE 2 -- Uploading video to Shotstack...")
    try:
        video_url = upload_video(api_key, video_path)
    except RuntimeError as exc:
        log.error(str(exc))
        sys.exit(2)

    # --- Stage 3: logo ---
    log.info("=" * 60)
    logo_uri = None
    if logo_path.exists():
        log.info("STAGE 3 -- Encoding logo.png...")
        logo_uri = encode_logo(logo_path)
        log.info(f"  logo.png encoded ({logo_path.stat().st_size / 1024:.1f} KB)")
    else:
        log.info("STAGE 3 -- logo.png not found, skipping watermark.")

    # --- Stage 4: captions ---
    log.info("=" * 60)
    captions = []
    if captions_path.exists():
        log.info("STAGE 4 -- Parsing captions.srt...")
        captions = parse_srt(captions_path)
        log.info(f"  Parsed {len(captions)} subtitle entries.")
    else:
        log.info("STAGE 4 -- captions.srt not found, skipping captions.")

    # --- Stage 5: build timelines ---
    log.info("=" * 60)
    log.info("STAGE 5 -- Building timelines...")
    yt_timeline   = build_timeline(video_url, duration, logo_uri, captions, 1920, 1080)
    reel_timeline = build_timeline(video_url, duration, logo_uri, captions, 1080, 1920)
    yt_tracks   = len(yt_timeline["timeline"]["tracks"])
    log.info(f"  YouTube  timeline: {yt_tracks} track(s), {duration:.1f}s")
    log.info(f"  Reels    timeline: {len(reel_timeline['timeline']['tracks'])} track(s), {duration:.1f}s")

    # --- Stage 6: submit renders ---
    log.info("=" * 60)
    log.info("STAGE 6 -- Submitting render jobs...")
    try:
        yt_id   = submit_render(api_key, yt_timeline,   "16:9", "YouTube")
        reel_id = submit_render(api_key, reel_timeline, "9:16", "Reels")
    except RuntimeError as exc:
        log.error(str(exc))
        sys.exit(2)

    # --- Stage 7: poll ---
    log.info("=" * 60)
    log.info(f"STAGE 7 -- Polling renders every {POLL_INTERVAL}s...")
    try:
        yt_url, reel_url = poll_both(api_key, yt_id, reel_id)
    except RuntimeError as exc:
        log.error(str(exc))
        sys.exit(2)

    # --- Stage 8: download ---
    log.info("=" * 60)
    log.info("STAGE 8 -- Downloading final renders...")
    try:
        download_file(yt_url,   yt_out)
        download_file(reel_url, reel_out)
    except requests.RequestException as exc:
        log.error("Download failed: %s", exc)
        sys.exit(2)

    log.info("=" * 60)
    log.info("Done.")
    log.info(f"  final_youtube.mp4 -> {yt_out.resolve()}")
    log.info(f"  final_reel.mp4    -> {reel_out.resolve()}")


if __name__ == "__main__":
    main()
