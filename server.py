import os
import sys
import uuid
import json
import shutil
import logging
import tempfile
import threading
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from typing import List

import requests
import boto3
from botocore.config import Config
from flask import Flask, request, jsonify

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

app = Flask(__name__)

SCRIPT_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# In-memory job store (single-worker gunicorn + threads shares one process)
# ---------------------------------------------------------------------------

_jobs: dict = {}
_lock = threading.Lock()

OUTPUT_FILES = [
    "gemini_output.json",
    "enhanced_video.mp4",
    "captions.srt",
    "transcript.txt",
    "reel_cut.mp4",
    "youtube_cut.mp4",
    "graded_video.mp4",
    "final_youtube.mp4",
    "final_reel.mp4",
    "job_log.json",
]


# ---------------------------------------------------------------------------
# R2 client
# ---------------------------------------------------------------------------

def _r2_client():
    account_id = os.environ["R2_ACCOUNT_ID"]
    return boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )


def _upload_outputs(job_id: str, work_dir: Path) -> dict:
    """Upload all produced files to R2; return {filename: url}."""
    bucket = os.environ.get("R2_BUCKET", "oneraw-studio")
    client = _r2_client()
    urls = {}

    for name in OUTPUT_FILES:
        p = work_dir / name
        if not p.exists():
            continue
        key = f"{job_id}/{name}"
        log.info("Uploading %s -> s3://%s/%s", name, bucket, key)
        client.upload_file(str(p), bucket, key)

        # Generate a presigned URL valid for 7 days (use public URL if configured)
        public_base = os.environ.get("R2_PUBLIC_URL", "").rstrip("/")
        if public_base:
            urls[name] = f"{public_base}/{key}"
        else:
            urls[name] = client.generate_presigned_url(
                "get_object",
                Params={"Bucket": bucket, "Key": key},
                ExpiresIn=604800,
            )

    return urls


# ---------------------------------------------------------------------------
# Job state helpers
# ---------------------------------------------------------------------------

def _set_state(job_id: str, **kwargs):
    with _lock:
        _jobs[job_id].update(kwargs)


def _get_job(job_id: str) -> dict | None:
    with _lock:
        return dict(_jobs.get(job_id, {}))


# ---------------------------------------------------------------------------
# Download + merge helpers
# ---------------------------------------------------------------------------

CROSSFADE_DURATION = 1.0  # seconds — overlap between consecutive clips


def _download_video(url: str, dest: Path) -> None:
    with requests.get(url, stream=True, timeout=600) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)


def _get_duration(path: Path) -> float:
    """Return video duration in seconds via ffprobe."""
    # Try stream-level duration first (more accurate for VFR/containers)
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=duration",
        "-of", "json",
        str(path),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode == 0:
        streams = json.loads(result.stdout).get("streams", [])
        if streams and streams[0].get("duration"):
            return float(streams[0]["duration"])

    # Fallback: container-level duration
    cmd2 = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        str(path),
    ]
    result2 = subprocess.run(cmd2, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result2.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path.name}: {result2.stderr}")
    dur = json.loads(result2.stdout).get("format", {}).get("duration")
    if not dur:
        raise RuntimeError(f"No duration found for {path.name}")
    return float(dur)


def _merge_videos(paths: List[Path], output: Path) -> None:
    """Concatenate clips with xfade/acrossfade crossfade transitions.

    Each consecutive pair overlaps by CROSSFADE_DURATION seconds.
    xfade offsets are accumulated across the chain:
        offset_i = sum(d[0..i-1]) - i * xd
    """
    n = len(paths)

    # Probe durations
    durations: List[float] = []
    for p in paths:
        d = _get_duration(p)
        durations.append(d)
        log.info("  %s  %.2fs", p.name, d)

    # Clamp crossfade so it never exceeds half the shortest clip
    xd = min(CROSSFADE_DURATION, min(durations) / 2)
    log.info("Crossfade duration: %.2fs", xd)

    # Build -i flags
    inputs: List[str] = []
    for p in paths:
        inputs += ["-i", str(p)]

    # Build chained xfade (video) and acrossfade (audio) filter graph.
    # After each xfade the running clock shrinks by xd because the clips overlap.
    v_filters: List[str] = []
    a_filters: List[str] = []
    current_v    = "[0:v]"
    current_a    = "[0:a]"
    running_dur  = durations[0]

    for i in range(1, n):
        offset = max(0.0, running_dur - xd)
        is_last = (i == n - 1)
        out_v   = "[v]"       if is_last else f"[xfv{i}]"
        out_a   = "[a]"       if is_last else f"[xfa{i}]"

        v_filters.append(
            f"{current_v}[{i}:v]"
            f"xfade=transition=fade:duration={xd:.3f}:offset={offset:.3f}"
            f"{out_v}"
        )
        a_filters.append(
            f"{current_a}[{i}:a]"
            f"acrossfade=d={xd:.3f}"
            f"{out_a}"
        )

        current_v    = out_v
        current_a    = out_a
        running_dur  = running_dur + durations[i] - xd

    filter_complex = ";".join(v_filters + a_filters)

    cmd = (
        ["ffmpeg", "-y"]
        + inputs
        + ["-filter_complex", filter_complex,
           "-map", "[v]",
           "-map", "[a]",
           "-c:v", "libx264", "-crf", "18", "-preset", "medium",
           "-c:a", "aac", "-b:a", "192k",
           "-movflags", "+faststart",
           str(output)]
    )

    log.info("Merging %d clips with %.1fs crossfade -> %s", n, xd, output.name)
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg crossfade merge failed:\n{result.stderr[-3000:]}")


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

def _run_job(job_id: str, video_urls: List[str]):
    work_dir = Path(tempfile.mkdtemp(prefix=f"oneraw_{job_id}_"))
    log.info("[%s] work_dir: %s", job_id, work_dir)

    try:
        # --- download -------------------------------------------------------
        _set_state(job_id, status="downloading")

        downloaded: List[Path] = []
        for i, url in enumerate(video_urls):
            log.info("[%s] Downloading video %d/%d: %s", job_id, i + 1, len(video_urls), url)
            name = url.split("?")[0].rstrip("/").split("/")[-1] or f"input_{i}.mp4"
            if not name.lower().endswith(".mp4"):
                name += ".mp4"
            # avoid collisions when two URLs have the same filename
            dest = work_dir / f"{i:02d}_{name}"
            _download_video(url, dest)
            size_mb = dest.stat().st_size / (1024 * 1024)
            log.info("[%s] Downloaded %.1f MB -> %s", job_id, size_mb, dest.name)
            downloaded.append(dest)

        # --- merge (only when more than one clip) ---------------------------
        if len(downloaded) == 1:
            video_path = downloaded[0]
        else:
            video_path = work_dir / "merged_input.mp4"
            try:
                _merge_videos(downloaded, video_path)
            except RuntimeError as exc:
                log.error("[%s] Merge failed: %s", job_id, exc)
                _set_state(
                    job_id,
                    status="failed",
                    error=str(exc),
                    finished_at=datetime.now(timezone.utc).isoformat(),
                )
                return
            size_mb = video_path.stat().st_size / (1024 * 1024)
            log.info("[%s] Merged -> %s  (%.1f MB)", job_id, video_path.name, size_mb)

        _set_state(job_id, video_path=str(video_path))

        # --- pipeline -------------------------------------------------------
        _set_state(job_id, status="running")
        pipeline_script = SCRIPT_DIR / "pipeline.py"
        cmd = [sys.executable, str(pipeline_script), str(video_path)]

        log.info("[%s] Launching pipeline: %s", job_id, " ".join(cmd))
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(work_dir),
            env=os.environ.copy(),
        )

        output_lines = []
        for line in proc.stdout:
            sys.stdout.write(f"[{job_id[:8]}] {line}")
            sys.stdout.flush()
            output_lines.append(line)

        proc.wait()
        exit_code = proc.returncode

        # read job_log produced by pipeline.py
        pipeline_log = {}
        job_log_path = work_dir / "job_log.json"
        if job_log_path.exists():
            try:
                pipeline_log = json.loads(job_log_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        if exit_code != 0:
            _set_state(
                job_id,
                status="failed",
                error=f"pipeline.py exited with code {exit_code}",
                pipeline_log=pipeline_log,
                finished_at=datetime.now(timezone.utc).isoformat(),
            )
            return

        # --- upload ---------------------------------------------------------
        _set_state(job_id, status="uploading")
        try:
            urls = _upload_outputs(job_id, work_dir)
        except Exception as exc:
            log.error("[%s] R2 upload failed: %s", job_id, exc)
            _set_state(
                job_id,
                status="failed",
                error=f"R2 upload failed: {exc}",
                pipeline_log=pipeline_log,
                finished_at=datetime.now(timezone.utc).isoformat(),
            )
            return

        _set_state(
            job_id,
            status="completed",
            outputs=urls,
            pipeline_log=pipeline_log,
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
        log.info("[%s] Completed. %d output(s) uploaded.", job_id, len(urls))

    except Exception as exc:
        log.exception("[%s] Unexpected error: %s", job_id, exc)
        _set_state(
            job_id,
            status="failed",
            error=str(exc),
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
    finally:
        try:
            shutil.rmtree(work_dir, ignore_errors=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/jobs", methods=["GET"])
def list_jobs():
    with _lock:
        summary = [
            {
                "job_id": jid,
                "status": info.get("status"),
                "queued_at": info.get("queued_at"),
                "finished_at": info.get("finished_at"),
            }
            for jid, info in _jobs.items()
        ]
    return jsonify(summary)


@app.route("/status/<job_id>", methods=["GET"])
def job_status(job_id: str):
    job = _get_job(job_id)
    if not job:
        return jsonify({"error": "job not found"}), 404
    return jsonify(job)


@app.route("/run-pipeline", methods=["POST"])
def run_pipeline():
    body = request.get_json(silent=True) or {}

    # Accept either a single string or an array
    raw_single = body.get("video_url")
    raw_multi  = body.get("video_urls")

    if raw_multi is not None:
        if not isinstance(raw_multi, list):
            return jsonify({"error": "video_urls must be an array"}), 400
        video_urls = [u.strip() for u in raw_multi if isinstance(u, str) and u.strip()]
    elif raw_single is not None:
        url = str(raw_single).strip()
        video_urls = [url] if url else []
    else:
        video_urls = []

    if not video_urls:
        return jsonify({"error": "video_url or video_urls is required"}), 400

    job_id    = str(uuid.uuid4())
    queued_at = datetime.now(timezone.utc).isoformat()

    with _lock:
        _jobs[job_id] = {
            "job_id":       job_id,
            "status":       "queued",
            "video_urls":   video_urls,
            "queued_at":    queued_at,
            "finished_at":  None,
            "outputs":      {},
            "error":        None,
            "pipeline_log": {},
        }

    thread = threading.Thread(target=_run_job, args=(job_id, video_urls), daemon=True)
    thread.start()

    return jsonify({
        "job_id":      job_id,
        "status":      "queued",
        "video_count": len(video_urls),
        "status_url":  f"/status/{job_id}",
    }), 202


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
