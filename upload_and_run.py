import os
import sys
import time
import json
import logging
import argparse
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

try:
    import anthropic
except ImportError:
    print("ERROR: anthropic not installed. Run: pip install anthropic")
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

PIPELINE_BASE  = "https://oneraw-studio-production.up.railway.app"
CLAUDE_MODEL   = "claude-sonnet-4-20250514"
POLL_INTERVAL  = 30    # seconds
PRESIGN_EXPIRY = 86400  # 24 hours

# ---------------------------------------------------------------------------
# Defaults used when Claude can't determine a value
# ---------------------------------------------------------------------------

DEFAULTS = {
    "color_preset":    "warm",
    "duration_seconds": 90,
    "language":        "english",
    "music_mood":      "emotional",
    "output_format":   "both",
}

VALID = {
    "color_preset":   {"warm", "neutral", "moody", "vivid"},
    "language":       {"telugu", "hindi", "english"},
    "music_mood":     {"slow", "upbeat", "emotional", "energetic"},
    "output_format":  {"reel", "youtube", "both"},
}


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
# Stage 1: interpret prompt with Claude
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a video production assistant. The user describes a video they want produced.
Extract the following settings from their description and return ONLY a JSON object — no
explanation, no markdown, no code fences, just the raw JSON.

Fields to extract:
  color_preset    — one of: warm, neutral, moody, vivid
  duration_seconds — integer (seconds); use 90 for "reel" / "short", 480 for "youtube" / "long"
  language        — one of: telugu, hindi, english
  music_mood      — one of: slow, upbeat, emotional, energetic
  output_format   — one of: reel, youtube, both

When a field is ambiguous or not mentioned, pick the most sensible default for the
described content. Always return all five fields.
"""


def interpret_prompt(api_key: str, prompt: str) -> dict:
    log.info("=" * 60)
    log.info("STAGE 1 -- Interpreting prompt with Claude (%s)", CLAUDE_MODEL)
    log.info("  Prompt : %s", prompt)

    client = anthropic.Anthropic(api_key=api_key)

    message = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=256,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = message.content[0].text.strip()
    log.info("  Raw response : %s", raw)

    try:
        settings = json.loads(raw)
    except json.JSONDecodeError as exc:
        log.error("Claude returned non-JSON: %s  (%s)", raw, exc)
        raise RuntimeError("Failed to parse Claude response as JSON.") from exc

    # validate / coerce each field, fall back to defaults for unknown values
    result = {}

    for field, default in DEFAULTS.items():
        val = settings.get(field, default)
        allowed = VALID.get(field)

        if field == "duration_seconds":
            try:
                result[field] = max(10, int(val))
            except (TypeError, ValueError):
                log.warning("  Invalid duration '%s', using default %s", val, default)
                result[field] = default
        elif allowed and str(val).lower() not in allowed:
            log.warning("  Invalid %s '%s', using default '%s'", field, val, default)
            result[field] = default
        else:
            result[field] = str(val).lower() if allowed else val

    log.info("  Extracted settings:")
    for k, v in result.items():
        log.info("    %-20s %s", k, v)

    return result


# ---------------------------------------------------------------------------
# Stage 2: upload to R2
# ---------------------------------------------------------------------------

def upload_to_r2(client, bucket: str, video_path: Path) -> str:
    key  = f"raw-videos/{video_path.name}"
    size = video_path.stat().st_size

    log.info("=" * 60)
    log.info("STAGE 2 -- Uploading to R2")
    log.info("  Bucket : %s", bucket)
    log.info("  Key    : %s", key)
    log.info("  Size   : %s", fmt_size(size))

    client.upload_file(str(video_path), bucket, key)
    log.info("  Upload complete.")
    return key


# ---------------------------------------------------------------------------
# Stage 3: presign
# ---------------------------------------------------------------------------

def presign(client, bucket: str, key: str) -> str:
    log.info("=" * 60)
    log.info("STAGE 3 -- Generating presigned URL (24h)")

    url = client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=PRESIGN_EXPIRY,
    )
    log.info("  URL : %s", url[:80] + "..." if len(url) > 80 else url)
    return url


# ---------------------------------------------------------------------------
# Stage 4: trigger pipeline
# ---------------------------------------------------------------------------

def trigger_pipeline(video_url: str, settings: dict) -> str:
    log.info("=" * 60)
    log.info("STAGE 4 -- Triggering pipeline")

    payload = {"video_url": video_url, **settings}
    log.info("  POST %s/run-pipeline", PIPELINE_BASE)
    log.info("  Body : %s", json.dumps(payload))

    resp = requests.post(
        f"{PIPELINE_BASE}/run-pipeline",
        json=payload,
        timeout=30,
    )

    if not resp.ok:
        log.error("Pipeline trigger failed (%s): %s", resp.status_code, resp.text)
        raise RuntimeError("Failed to trigger pipeline.")

    body   = resp.json()
    job_id = body.get("job_id")
    if not job_id:
        log.error("Unexpected response: %s", body)
        raise RuntimeError("No job_id in pipeline response.")

    log.info("  Job ID : %s", job_id)
    log.info("  Status : %s", body.get("status"))
    return job_id


# ---------------------------------------------------------------------------
# Stage 5: poll
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
    url = f"{PIPELINE_BASE}/status/{job_id}"
    t0  = time.perf_counter()

    log.info("=" * 60)
    log.info("STAGE 5 -- Polling every %ds", POLL_INTERVAL)
    log.info("  Status URL : %s", url)

    while True:
        try:
            resp = requests.get(url, timeout=30)
        except requests.RequestException as exc:
            log.warning("  Poll request failed (%s) — will retry", exc)
            time.sleep(POLL_INTERVAL)
            continue

        if not resp.ok:
            log.warning("  Poll returned %s — will retry", resp.status_code)
            time.sleep(POLL_INTERVAL)
            continue

        job   = resp.json()
        state = job.get("status", "unknown")
        label = STATUS_LABELS.get(state, state)
        wall  = fmt_elapsed(time.perf_counter() - t0)

        log.info("  [%s elapsed]  %s", wall, label)

        if state == "failed":
            err          = job.get("error", "")
            failed_stage = job.get("pipeline_log", {}).get("failed_stage", "")
            if failed_stage:
                log.error("  Failed stage : %s", failed_stage)
            if err:
                log.error("  Error        : %s", err)

        if state in TERMINAL_STATES:
            return job

        time.sleep(POLL_INTERVAL)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload a video to R2, interpret a creative prompt with Claude, "
                    "then run the OneRaw pipeline and stream progress.",
    )
    parser.add_argument("video", help="Path to the local video file")
    parser.add_argument(
        "--prompt", "-p",
        required=True,
        help=(
            'Natural-language description of the desired output, e.g. '
            '"wedding reel, warm color, telugu captions, emotional music, 90 seconds"'
        ),
    )
    args = parser.parse_args()

    try:
        video_path = validate_input(args.video)
    except (FileNotFoundError, ValueError) as exc:
        log.error(str(exc))
        sys.exit(1)

    anthropic_key = get_env("ANTHROPIC_API_KEY")
    account_id    = get_env("R2_ACCOUNT_ID")
    access_key    = get_env("R2_ACCESS_KEY_ID")
    secret_key    = get_env("R2_SECRET_ACCESS_KEY")
    bucket        = os.environ.get("R2_BUCKET", "raw-videos").strip()

    s3 = r2_client(account_id, access_key, secret_key)

    log.info("=" * 60)
    log.info("  OneRaw upload-and-run")
    log.info("  Video  : %s  (%s)", video_path.name, fmt_size(video_path.stat().st_size))
    log.info("  Prompt : %s", args.prompt)
    log.info("=" * 60)

    # 1 — interpret prompt
    try:
        settings = interpret_prompt(anthropic_key, args.prompt)
    except RuntimeError as exc:
        log.error(str(exc))
        sys.exit(2)

    # 2 — upload
    try:
        key = upload_to_r2(s3, bucket, video_path)
    except Exception as exc:
        log.error("R2 upload failed: %s", exc)
        sys.exit(2)

    # 3 — presign
    try:
        presigned_url = presign(s3, bucket, key)
    except Exception as exc:
        log.error("Presign failed: %s", exc)
        sys.exit(2)

    # 4 — trigger
    try:
        job_id = trigger_pipeline(presigned_url, settings)
    except RuntimeError as exc:
        log.error(str(exc))
        sys.exit(2)

    print(f"\nJob ID: {job_id}\n")

    # 5 — poll
    job = poll_job(job_id)

    # 6 — results
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
