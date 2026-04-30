import sys
import os
import time
import json
import uuid
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent
DIVIDER    = "=" * 64


# ---------------------------------------------------------------------------
# Stage runner — streams output to console and captures for the log
# ---------------------------------------------------------------------------

def _safe_write(line: str) -> None:
    """Write a line to stdout, replacing characters the console can't encode."""
    try:
        sys.stdout.write(line)
    except UnicodeEncodeError:
        enc  = sys.stdout.encoding or "ascii"
        safe = line.encode(enc, errors="replace").decode(enc)
        sys.stdout.write(safe)
    sys.stdout.flush()


def run_stage(name: str, script: str, args: list, cwd: Path) -> dict:
    script_path = SCRIPT_DIR / script
    started_at  = datetime.now(timezone.utc)
    t0          = time.perf_counter()

    if not script_path.exists():
        return {
            "name":             name,
            "script":           script,
            "args":             args,
            "status":           "failed",
            "exit_code":        -1,
            "started_at":       started_at.isoformat(),
            "finished_at":      datetime.now(timezone.utc).isoformat(),
            "duration_seconds": 0.0,
            "output":           f"Script not found: {script_path}",
        }

    cmd = [sys.executable, str(script_path)] + [str(a) for a in args]

    # --- launch (isolated try so a launch failure doesn't mask stream errors) ---
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(cwd),
            env=os.environ.copy(),
        )
    except Exception as exc:
        duration = time.perf_counter() - t0
        return {
            "name":             name,
            "script":           script,
            "args":             args,
            "status":           "failed",
            "exit_code":        -1,
            "started_at":       started_at.isoformat(),
            "finished_at":      datetime.now(timezone.utc).isoformat(),
            "duration_seconds": round(duration, 2),
            "output":           f"Failed to launch subprocess: {exc}",
        }

    # --- stream output to console + buffer for log ---
    output_buf = []
    for line in proc.stdout:
        _safe_write(line)
        output_buf.append(line)

    proc.wait()
    exit_code   = proc.returncode
    duration    = time.perf_counter() - t0
    finished_at = datetime.now(timezone.utc)
    full_output = "".join(output_buf)

    return {
        "name":             name,
        "script":           script,
        "args":             args,
        "status":           "completed" if exit_code == 0 else "failed",
        "exit_code":        exit_code,
        "started_at":       started_at.isoformat(),
        "finished_at":      finished_at.isoformat(),
        "duration_seconds": round(duration, 2),
        # keep last 8 000 chars so the log stays readable for long runs
        "output":           full_output[-8000:] if len(full_output) > 8000 else full_output,
    }


# ---------------------------------------------------------------------------
# Pipeline definition
# Relative file paths are resolved against CWD (where outputs land).
# ---------------------------------------------------------------------------

def build_pipeline(input_video: Path) -> list:
    v = str(input_video)           # original video (absolute)
    return [
        {
            "name":   "Gemini Analyze",
            "script": "gemini_analyze.py",
            "args":   [v],
            "produces": ["gemini_output.json"],
        },
        {
            "name":   "Dolby Enhance",
            "script": "dolby_enhance.py",
            "args":   [v],
            "produces": ["enhanced_video.mp4"],
        },
        {
            "name":   "Gladia Captions",
            "script": "gladia_captions.py",
            "args":   ["enhanced_video.mp4"],
            "produces": ["captions.srt", "transcript.txt"],
        },
        {
            "name":   "FFmpeg Cut",
            "script": "ffmpeg_cut.py",
            "args":   ["enhanced_video.mp4", "gemini_output.json"],
            "produces": ["reel_cut.mp4", "youtube_cut.mp4"],
        },
        {
            "name":   "Color Grade",
            "script": "colorgrade.py",
            "args":   ["reel_cut.mp4", "--preset", "warm", "-o", "graded_video.mp4"],
            "produces": ["graded_video.mp4"],
        },
        {
            "name":   "Shotstack Render",
            "script": "shotstack_render.py",
            "args":   ["graded_video.mp4"],
            "produces": ["final_youtube.mp4", "final_reel.mp4"],
        },
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s"


def save_log(path: Path, job: dict) -> None:
    path.write_text(
        json.dumps(job, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: python {Path(__file__).name} <input_video>")
        sys.exit(1)

    input_video = Path(sys.argv[1]).resolve()
    if not input_video.exists():
        log.error(f"Input video not found: {input_video}")
        sys.exit(1)

    job_id   = str(uuid.uuid4())
    cwd      = Path.cwd()
    log_path = cwd / "job_log.json"
    pipeline = build_pipeline(input_video)

    # ── job manifest (written incrementally so crashes still leave a log) ──
    job = {
        "job_id":                job_id,
        "input_video":           str(input_video),
        "started_at":            datetime.now(timezone.utc).isoformat(),
        "finished_at":           None,
        "total_duration_seconds": None,
        "status":                "running",
        "failed_stage":          None,
        "stages":                [],
    }

    # ── header ──────────────────────────────────────────────────────────────
    log.info(DIVIDER)
    log.info(f"  PIPELINE START")
    log.info(f"  Job ID : {job_id}")
    log.info(f"  Input  : {input_video}")
    log.info(f"  Stages : {len(pipeline)}")
    log.info(DIVIDER)

    t_pipeline_start = time.perf_counter()
    failed           = False

    for i, stage_def in enumerate(pipeline, start=1):
        name     = stage_def["name"]
        script   = stage_def["script"]
        args     = stage_def["args"]
        produces = stage_def.get("produces", [])

        log.info("")
        log.info(DIVIDER)
        log.info(f"  [{i}/{len(pipeline)}]  {name.upper()}")
        log.info(f"  Script : {script}")
        log.info(f"  Args   : {' '.join(str(a) for a in args)}")
        log.info(DIVIDER)

        # ── skip if all output files already exist ───────────────────────────
        existing = [f for f in produces if (cwd / f).exists()]
        if produces and len(existing) == len(produces):
            log.info(f"  [SKIP] Outputs already exist: {', '.join(existing)}")
            job["stages"].append({
                "name":             name,
                "script":           script,
                "args":             args,
                "status":           "skipped",
                "exit_code":        0,
                "started_at":       datetime.now(timezone.utc).isoformat(),
                "finished_at":      datetime.now(timezone.utc).isoformat(),
                "duration_seconds": 0.0,
                "output":           f"Skipped — outputs already present: {existing}",
            })
            save_log(log_path, job)
            continue

        result = run_stage(name, script, args, cwd)
        job["stages"].append(result)

        # persist log after every stage so a crash leaves a partial record
        save_log(log_path, job)

        dur = fmt_duration(result["duration_seconds"])
        log.info("")
        if result["status"] == "completed":
            log.info(f"  [DONE]  {name}  ({dur})")
        else:
            log.error(f"  [FAIL]  {name}  (exit code {result['exit_code']},  {dur})")
            job["status"]       = "failed"
            job["failed_stage"] = name
            failed = True
            break

    total_duration          = time.perf_counter() - t_pipeline_start
    job["finished_at"]      = datetime.now(timezone.utc).isoformat()
    job["total_duration_seconds"] = round(total_duration, 2)
    if not failed:
        job["status"] = "completed"

    save_log(log_path, job)

    # ── summary ─────────────────────────────────────────────────────────────
    log.info("")
    log.info(DIVIDER)
    if not failed:
        log.info(f"  PIPELINE COMPLETE")
        log.info(f"  Total time : {fmt_duration(total_duration)}")
        log.info("")
        log.info("  Stage breakdown:")
        for s in job["stages"]:
            log.info(f"    {s['name']:<22} {fmt_duration(s['duration_seconds'])}")
        log.info("")
        log.info("  Outputs:")
        for stage_def in pipeline:
            for f in stage_def.get("produces", []):
                p = cwd / f
                size = f"({p.stat().st_size / (1024*1024):.1f} MB)" if p.exists() else "(missing)"
                log.info(f"    {f:<30} {size}")
    else:
        log.error(f"  PIPELINE FAILED at: {job['failed_stage']}")
        log.info(f"  Ran for : {fmt_duration(total_duration)}")
        completed = [s["name"] for s in job["stages"] if s["status"] == "completed"]
        if completed:
            log.info(f"  Completed stages: {', '.join(completed)}")

    log.info("")
    log.info(f"  Job log : {log_path}")
    log.info(DIVIDER)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
