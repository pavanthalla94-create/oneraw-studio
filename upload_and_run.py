import os
import sys
import time
import logging
from pathlib import Path

try:
    import boto3
    from botocore.config import Config
except ImportError:
    print("ERROR: boto3 not installed. Run: pip install boto3")
    sys.exit(1)

try:
    import requests
except ImportError:
    print("ERROR: requests not installed. Run: pip install requests")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

PIPELINE_BASE  = "https://oneraw-studio-production.up.railway.app"
POLL_INTERVAL  = 30   # seconds
PRESIGN_EXPIRY = 86400  # 24 hours


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_env(name: str) -> str:
    val = os.environ.get(name, "").strip()
    if not val:
        log.error("Environment variable %s is not set.", name)
        sys.exit(1)
    return val


def r2_client(account_id: str, access_key: str, secret_key: str):
    return boto3.client(
        "s3",
        endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )


def validate_input(path: str) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    if not p.is_file():
        raise ValueError(f"Not a file: {p}")
    return p


def fmt_size(n_bytes: int) -> str:
    if n_bytes >= 1_073_741_824:
        return f"{n_bytes / 1_073_741_824:.1f} GB"
    if n_bytes >= 1_048_576:
        return f"{n_bytes / 1_048_576:.1f} MB"
    return f"{n_bytes / 1024:.1f} KB"


def fmt_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{int(seconds)}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s"


# ---------------------------------------------------------------------------
# Stage 1: upload to R2
# ---------------------------------------------------------------------------

def upload_to_r2(client, bucket: str, video_path: Path) -> str:
    key = f"raw-videos/{video_path.name}"
    size = video_path.stat().st_size

    log.info("=" * 60)
    log.info("STAGE 1 -- Uploading to R2")
    log.info("  Bucket : %s", bucket)
    log.info("  Key    : %s", key)
    log.info("  Size   : %s", fmt_size(size))

    client.upload_file(str(video_path), bucket, key)
    log.info("  Upload complete.")
    return key


# ---------------------------------------------------------------------------
# Stage 2: presign
# ---------------------------------------------------------------------------

def presign(client, bucket: str, key: str) -> str:
    log.info("=" * 60)
    log.info("STAGE 2 -- Generating presigned URL (24h)")

    url = client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=PRESIGN_EXPIRY,
    )
    log.info("  URL    : %s", url[:80] + "..." if len(url) > 80 else url)
    return url


# ---------------------------------------------------------------------------
# Stage 3: trigger pipeline
# ---------------------------------------------------------------------------

def trigger_pipeline(video_url: str) -> str:
    log.info("=" * 60)
    log.info("STAGE 3 -- Triggering pipeline")
    log.info("  POST %s/run-pipeline", PIPELINE_BASE)

    resp = requests.post(
        f"{PIPELINE_BASE}/run-pipeline",
        json={"video_url": video_url},
        timeout=30,
    )

    if not resp.ok:
        log.error("Pipeline trigger failed (%s): %s", resp.status_code, resp.text)
        raise RuntimeError("Failed to trigger pipeline.")

    body = resp.json()
    job_id = body.get("job_id")
    if not job_id:
        log.error("Unexpected response: %s", body)
        raise RuntimeError("No job_id in pipeline response.")

    log.info("  Job ID : %s", job_id)
    log.info("  Status : %s", body.get("status"))
    log.info("  Poll   : %s/status/%s", PIPELINE_BASE, job_id)
    return job_id


# ---------------------------------------------------------------------------
# Stage 4: poll
# ---------------------------------------------------------------------------

TERMINAL_STATES = {"completed", "failed"}

STATUS_LABELS = {
    "queued":      "Queued — waiting to start",
    "downloading": "Downloading video...",
    "running":     "Pipeline running...",
    "uploading":   "Uploading outputs to R2...",
    "completed":   "Completed",
    "failed":      "Failed",
}


def poll_job(job_id: str) -> dict:
    url     = f"{PIPELINE_BASE}/status/{job_id}"
    elapsed = 0
    t0      = time.perf_counter()

    log.info("=" * 60)
    log.info("STAGE 4 -- Polling every %ds", POLL_INTERVAL)

    while True:
        try:
            resp = requests.get(url, timeout=30)
        except requests.RequestException as exc:
            log.warning("  Poll request failed (%s) — will retry", exc)
            time.sleep(POLL_INTERVAL)
            elapsed += POLL_INTERVAL
            continue

        if not resp.ok:
            log.warning("  Poll returned %s — will retry", resp.status_code)
            time.sleep(POLL_INTERVAL)
            elapsed += POLL_INTERVAL
            continue

        job   = resp.json()
        state = job.get("status", "unknown")
        label = STATUS_LABELS.get(state, state)
        wall  = fmt_elapsed(time.perf_counter() - t0)

        log.info("  [%s elapsed]  %s", wall, label)

        # surface the failed stage from pipeline_log if available
        if state == "failed":
            err = job.get("error", "")
            pl  = job.get("pipeline_log", {})
            failed_stage = pl.get("failed_stage", "")
            if failed_stage:
                log.error("  Failed stage : %s", failed_stage)
            if err:
                log.error("  Error        : %s", err)

        if state in TERMINAL_STATES:
            return job

        time.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: python {Path(__file__).name} <video_file>")
        sys.exit(1)

    try:
        video_path = validate_input(sys.argv[1])
    except (FileNotFoundError, ValueError) as exc:
        log.error(str(exc))
        sys.exit(1)

    account_id = get_env("R2_ACCOUNT_ID")
    access_key = get_env("R2_ACCESS_KEY_ID")
    secret_key = get_env("R2_SECRET_ACCESS_KEY")
    bucket     = os.environ.get("R2_BUCKET", "raw-videos").strip()

    client = r2_client(account_id, access_key, secret_key)

    log.info("=" * 60)
    log.info("  OneRaw upload-and-run")
    log.info("  Video  : %s", video_path)
    log.info("  Bucket : %s", bucket)
    log.info("=" * 60)

    # 1 — upload
    try:
        key = upload_to_r2(client, bucket, video_path)
    except Exception as exc:
        log.error("R2 upload failed: %s", exc)
        sys.exit(2)

    # 2 — presign
    try:
        presigned_url = presign(client, bucket, key)
    except Exception as exc:
        log.error("Presign failed: %s", exc)
        sys.exit(2)

    # 3 — trigger
    try:
        job_id = trigger_pipeline(presigned_url)
    except RuntimeError as exc:
        log.error(str(exc))
        sys.exit(2)

    print(f"\nJob ID: {job_id}\n")

    # 4 — poll
    job = poll_job(job_id)

    # 5 — results
    log.info("=" * 60)
    if job.get("status") == "completed":
        outputs = job.get("outputs", {})
        if outputs:
            log.info("  OUTPUT FILES")
            log.info("=" * 60)
            for name, url in outputs.items():
                log.info("  %-30s", name)
                log.info("    %s", url)
        else:
            log.warning("  Job completed but no output URLs returned.")
        log.info("=" * 60)
    else:
        log.error("  PIPELINE FAILED")
        log.info("=" * 60)
        sys.exit(3)


if __name__ == "__main__":
    main()
