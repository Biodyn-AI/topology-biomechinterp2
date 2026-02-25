#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

PAPER_TEX_REL_PATH = Path("paper/autoloop_research_paper.tex")
PAPER_PDF_REL_PATH = Path("paper/autoloop_research_paper.pdf")

LOOP_GENERATED_EXECUTOR_FILES = {
    "executor_prompt.md",
    "executor_stdout.log",
    "executor_stderr.log",
    "executor_last_message.md",
    "executor_exception.txt",
    "executor_research_validation.json",
}

NARRATIVE_SUFFIXES = {".md", ".txt"}
CODE_SUFFIXES = {".py", ".sh", ".ipynb", ".r", ".jl"}
MACHINE_RESULTS_SUFFIXES = {
    ".json",
    ".csv",
    ".tsv",
    ".parquet",
    ".npy",
    ".npz",
    ".pkl",
    ".pt",
}


@dataclass
class LoopConfig:
    project_root: Path
    executor_prompt_file: Path
    brainstormer_prompt_template_file: Path
    brainstormer_reference_root: Path
    executor_workdir: Path
    codex_bin: str
    max_iterations: Optional[int]
    sleep_seconds: int
    model: Optional[str]
    test_mode: bool
    reasoning_effort: str
    codex_timeout_seconds: int


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True))
        f.write("\n")


def _next_iteration_number(iterations_dir: Path) -> int:
    max_found = 0
    for child in iterations_dir.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        if not name.startswith("iter_"):
            continue
        suffix = name[5:]
        if suffix.isdigit():
            max_found = max(max_found, int(suffix))
    return max_found + 1


def _iteration_dirs_sorted(iterations_dir: Path) -> list[Path]:
    iter_dirs: list[tuple[int, Path]] = []
    for child in iterations_dir.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        if not name.startswith("iter_"):
            continue
        suffix = name[5:]
        if not suffix.isdigit():
            continue
        iter_dirs.append((int(suffix), child))
    iter_dirs.sort(key=lambda t: t[0], reverse=True)
    return [path for _, path in iter_dirs]


def _recent_hypothesis_outcomes_text(
    iterations_dir: Path,
    current_iteration_number: int,
    max_iterations_to_scan: int = 8,
    max_lines: int = 24,
) -> str:
    lines: list[str] = []
    family_negative_counts: dict[str, int] = {}
    scanned = 0
    for iter_dir in _iteration_dirs_sorted(iterations_dir):
        if scanned >= max_iterations_to_scan:
            break
        name = iter_dir.name
        suffix = name[5:]
        if not suffix.isdigit():
            continue
        iter_no = int(suffix)
        if iter_no >= current_iteration_number:
            continue
        screen_path = iter_dir / "executor_hypothesis_screen.json"
        if not screen_path.exists():
            continue
        scanned += 1
        try:
            payload = json.loads(_read_text(screen_path))
        except Exception:
            continue
        hypotheses = payload.get("hypotheses", [])
        if not isinstance(hypotheses, list):
            continue
        for h in hypotheses:
            if len(lines) >= max_lines:
                break
            family = str(h.get("family", "unknown"))
            hid = str(h.get("id", "H?"))
            decision = str(h.get("decision", "unknown"))
            direction = str(h.get("result_direction", "unknown"))
            method = str(h.get("method", ""))[:90]
            lines.append(
                f"- {iter_dir.name} {hid} | family={family} | decision={decision} | direction={direction} | method={method}"
            )
            if decision in {"negative", "inconclusive"}:
                family_negative_counts[family] = family_negative_counts.get(family, 0) + 1
        if len(lines) >= max_lines:
            break

    retire_candidates = [fam for fam, n in sorted(family_negative_counts.items()) if n >= 2]
    if not lines:
        return "No prior hypothesis history available."

    retire_line = (
        ", ".join(retire_candidates)
        if retire_candidates
        else "none detected"
    )
    return (
        "Recent hypothesis outcomes (most recent first):\n"
        + "\n".join(lines)
        + "\nSuggested retirement candidates by repeated negatives/inconclusive: "
        + retire_line
    )


def _load_loop_config(config_path: Path, args: argparse.Namespace) -> LoopConfig:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return LoopConfig(
        project_root=Path(payload["project_root"]).resolve(),
        executor_prompt_file=Path(payload["executor_prompt_file"]).resolve(),
        brainstormer_prompt_template_file=Path(payload["brainstormer_prompt_template_file"]).resolve(),
        brainstormer_reference_root=Path(payload["brainstormer_reference_root"]).resolve(),
        executor_workdir=Path(payload["executor_workdir"]).resolve(),
        codex_bin=str(payload.get("codex_bin", "codex")),
        max_iterations=args.max_iterations,
        sleep_seconds=args.sleep_seconds,
        model=args.model,
        test_mode=args.test_mode,
        reasoning_effort=args.reasoning_effort,
        codex_timeout_seconds=args.codex_timeout_seconds,
    )


def _paper_tex_path(project_root: Path) -> Path:
    return project_root / PAPER_TEX_REL_PATH


def _paper_pdf_path(project_root: Path) -> Path:
    return project_root / PAPER_PDF_REL_PATH


def _is_appledouble(path: Path) -> bool:
    return path.name.startswith("._")


def _relative_files(iter_dir: Path) -> list[Path]:
    files: list[Path] = []
    for p in iter_dir.rglob("*"):
        if p.is_file():
            files.append(p.relative_to(iter_dir))
    return files


def _report_has_command_trace(report_path: Path) -> bool:
    if not report_path.exists():
        return False
    text = _read_text(report_path).lower()
    command_markers = [
        "python ",
        "python3 ",
        "bash ",
        "rscript ",
        "uv run ",
        "snakemake ",
        "pytest ",
        "make ",
        "command:",
        "reproduction",
    ]
    return any(marker in text for marker in command_markers)


def _validate_executor_research_progress(iter_dir: Path, project_root: Path, iteration_number: int) -> dict:
    rel_files = _relative_files(iter_dir)
    user_files = [
        rf
        for rf in rel_files
        if not _is_appledouble(rf) and rf.name not in LOOP_GENERATED_EXECUTOR_FILES
    ]

    files_with_suffix = [(rf, rf.suffix.lower()) for rf in user_files]
    has_code_artifact = any(sfx in CODE_SUFFIXES for _, sfx in files_with_suffix)
    has_machine_results_artifact = any(sfx in MACHINE_RESULTS_SUFFIXES for _, sfx in files_with_suffix)
    has_non_narrative_artifact = any(
        (sfx not in NARRATIVE_SUFFIXES) and (rf.name != "executor_iteration_report.md")
        for rf, sfx in files_with_suffix
    )
    report_has_command_trace = _report_has_command_trace(iter_dir / "executor_iteration_report.md")
    hypothesis_screen_file = iter_dir / "executor_hypothesis_screen.json"
    hypothesis_screen_exists = hypothesis_screen_file.exists()
    hypothesis_screen_valid = False
    hypothesis_screen_count = 0
    hypothesis_screen_error = ""
    if hypothesis_screen_exists:
        try:
            payload = json.loads(_read_text(hypothesis_screen_file))
            hypotheses = payload.get("hypotheses", [])
            if isinstance(hypotheses, list):
                hypothesis_screen_count = len(hypotheses)
                hypothesis_screen_valid = hypothesis_screen_count > 0
            else:
                hypothesis_screen_error = "field 'hypotheses' is not a list"
        except Exception as exc:  # pragma: no cover - defensive against malformed output
            hypothesis_screen_error = f"invalid JSON: {type(exc).__name__}: {exc}"
    paper_tex_file = _paper_tex_path(project_root)
    paper_pdf_file = _paper_pdf_path(project_root)
    paper_tex_exists = paper_tex_file.exists()
    paper_pdf_exists = paper_pdf_file.exists()
    paper_pdf_nonempty = paper_pdf_exists and paper_pdf_file.stat().st_size > 0
    paper_iteration_marker = f"ITERATION UPDATE: iter_{iteration_number:04d}"
    paper_has_iteration_marker = False
    if paper_tex_exists:
        paper_has_iteration_marker = paper_iteration_marker in _read_text(paper_tex_file)
    paper_pdf_is_fresh = False
    if paper_tex_exists and paper_pdf_exists:
        paper_pdf_is_fresh = paper_pdf_file.stat().st_mtime >= paper_tex_file.stat().st_mtime

    passed = (
        has_machine_results_artifact
        and report_has_command_trace
        and hypothesis_screen_valid
        and paper_tex_exists
        and paper_has_iteration_marker
        and paper_pdf_nonempty
        and paper_pdf_is_fresh
    )
    failed_checks: list[str] = []
    if not has_machine_results_artifact:
        failed_checks.append("missing machine-readable results artifact (json/csv/tsv/parquet/npy/npz/pkl/pt)")
    if not report_has_command_trace:
        failed_checks.append("executor_iteration_report.md lacks explicit command trace/reproduction markers")
    if not hypothesis_screen_exists:
        failed_checks.append("missing required executor_hypothesis_screen.json artifact")
    elif not hypothesis_screen_valid:
        detail = f" ({hypothesis_screen_error})" if hypothesis_screen_error else ""
        failed_checks.append(
            f"executor_hypothesis_screen.json invalid or empty (requires non-empty 'hypotheses' list){detail}"
        )
    if not paper_tex_exists:
        failed_checks.append(f"LaTeX paper file missing: {paper_tex_file}")
    if paper_tex_exists and not paper_has_iteration_marker:
        failed_checks.append(
            f"paper missing required iteration marker '{paper_iteration_marker}' in {paper_tex_file}"
        )
    if not paper_pdf_exists:
        failed_checks.append(f"paper PDF missing: {paper_pdf_file}")
    if paper_pdf_exists and not paper_pdf_nonempty:
        failed_checks.append(f"paper PDF is empty: {paper_pdf_file}")
    if paper_tex_exists and paper_pdf_exists and not paper_pdf_is_fresh:
        failed_checks.append(f"paper PDF is stale (older than TeX source): {paper_pdf_file}")

    return {
        "passed_min_research_gate": passed,
        "failed_checks": failed_checks,
        "has_code_artifact": has_code_artifact,
        "has_machine_results_artifact": has_machine_results_artifact,
        "has_non_narrative_artifact": has_non_narrative_artifact,
        "report_has_command_trace": report_has_command_trace,
        "hypothesis_screen_file": str(hypothesis_screen_file),
        "hypothesis_screen_exists": hypothesis_screen_exists,
        "hypothesis_screen_valid": hypothesis_screen_valid,
        "hypothesis_screen_count": hypothesis_screen_count,
        "hypothesis_screen_error": hypothesis_screen_error,
        "paper_tex_file": str(paper_tex_file),
        "paper_pdf_file": str(paper_pdf_file),
        "paper_tex_exists": paper_tex_exists,
        "paper_pdf_exists": paper_pdf_exists,
        "paper_pdf_nonempty": paper_pdf_nonempty,
        "paper_pdf_is_fresh": paper_pdf_is_fresh,
        "paper_has_iteration_marker": paper_has_iteration_marker,
        "paper_iteration_marker": paper_iteration_marker,
        "detected_user_files": [str(p) for p in sorted(user_files)],
    }


def _build_executor_prompt(
    base_prompt_text: str,
    executor_prompt_file: Path,
    project_root: Path,
    iteration_dir: Path,
    iteration_number: int,
    brainstormer_feedback_text: str,
    recent_hypothesis_outcomes_text: str,
    test_mode: bool,
) -> str:
    # Keep this prompt explicit and deterministic so each iteration has a clear
    # contract: do real work, write artifacts, and consume brainstorming guidance.
    test_block = ""
    if test_mode:
        test_block = """
TEST MODE (for infrastructure validation):
- Keep this iteration lightweight and finish quickly.
- Do not run long experiments.
- Create/update only the required iteration artifacts and a brief realistic plan of what you would run in full mode.
- Budget: no more than 6 shell commands.
"""

    if test_mode:
        role_block = (
            "=== EXECUTOR ROLE PROMPT REFERENCE (TEST MODE) ===\n"
            f"Source file: {executor_prompt_file}\n"
            "Using a short excerpt for fast smoke validation. Full mode uses complete prompt.\n\n"
            + base_prompt_text[:1200]
            + "\n=== END EXCERPT ==="
        )
    else:
        role_block = (
            "=== EXECUTOR ROLE PROMPT START ===\n"
            + base_prompt_text
            + "\n=== EXECUTOR ROLE PROMPT END ==="
        )

    return f"""You are running as EXECUTOR in a persistent autonomous loop.

First, follow the role prompt below exactly.

{role_block}

Loop metadata:
- iteration_number: {iteration_number}
- project_root: {project_root}
- iteration_dir: {iteration_dir}

Mission:
Discover geometric/topological structures in scGPT + Geneformer residual representations by screening a broad hypothesis space.
{test_block}

Requirements for this iteration:
1. Perform concrete research progress (implementation and/or experiments), not just planning.
   - Run at least one explicit geometric/topological hypothesis test this iteration.
   - Prefer breadth-oriented screening: try new hypotheses or materially different variants, not only narrow polishing.
   - Avoid repeating directions that were already negative/inconclusive unless you provide a specific rescue rationale and changed method.
2. Write iteration artifacts directly inside:
   - {iteration_dir}
3. Required files to write:
   - {iteration_dir / "executor_iteration_report.md"}
   - {iteration_dir / "executor_next_steps.md"}
   - {iteration_dir / "executor_hypothesis_screen.json"}
4. Update cumulative project log:
   - {project_root / "reports" / "autoloop_master_log.md"}
5. Keep outputs reproducible (commands, paths, metrics, assumptions).
6. If blocked, document exact blocker and propose fallback action in the report.
7. Do not submit a paper-only or prose-only iteration.
8. Minimum research gate (mandatory):
   - produce at least one machine-readable results artifact generated this iteration (`.json`/`.csv`/`.tsv`/`.parquet`/`.npy`/`.npz`/`.pkl`/`.pt`),
   - include explicit command trace in `executor_iteration_report.md` (commands used to generate results),
   - ensure the report contains quantitative metrics derived from those artifacts,
   - populate `executor_hypothesis_screen.json` with a non-empty `hypotheses` list.
9. Hypothesis screen schema (strict):
   - write JSON object with field `hypotheses` as a list of entries.
   - each entry must contain:
     `id`, `name`, `family`, `split_regime`, `method`, `status`, `primary_metric`, `result_value`, `result_direction`, `artifact_paths`, `decision`, `next_action`.
   - `status` must be one of: `tested`, `partial`, `blocked`.
   - `decision` must be one of: `promising`, `neutral`, `negative`, `inconclusive`.
10. Focus areas to screen:
    - manifold geometry (geodesic neighborhoods, local linearity, intrinsic dimensionality),
    - topology (persistent homology / Betti summaries / graph-topology surrogates),
    - cross-model geometric consistency (scGPT vs Geneformer),
    - controls against shuffle/null/co-expression baselines.
11. Prefer code + analysis progression over narrative polishing. Paper text edits are allowed only as a secondary side effect.
12. Maintain the cumulative paper sources at:
   - {project_root / PAPER_TEX_REL_PATH}
   and compiled PDF at:
   - {project_root / PAPER_PDF_REL_PATH}
13. Paper update is mandatory every iteration:
   - if missing, create the LaTeX paper file;
   - if present, update it with new evidence;
   - include an explicit section header line exactly:
     `ITERATION UPDATE: iter_{iteration_number:04d}`
   - include quantitative results and direct artifact paths backing each new claim.
14. Compile LaTeX to PDF every iteration and ensure PDF reflects latest TeX changes.
    - Use reproducible compile commands (e.g., `latexmk -pdf` or `pdflatex`), and report them.

Hard stop condition for this Codex run:
- Execute exactly one iteration and then stop.
- After writing required files, return a concise final summary and terminate.
- Do not start another internal loop.

Recent hypothesis history from prior loop iterations (use it to avoid stale repeats):
{recent_hypothesis_outcomes_text}

Brainstormer guidance from previous iteration (must be addressed):
{brainstormer_feedback_text if brainstormer_feedback_text.strip() else "No brainstormer guidance yet; this is the first iteration."}
"""


def _build_brainstormer_prompt(
    brainstormer_template: str,
    project_root: Path,
    iteration_dir: Path,
    brainstormer_reference_root: Path,
) -> str:
    prompt = brainstormer_template.replace("{{ITERATION_DIR}}", str(iteration_dir))
    prompt = prompt.replace("{{PROJECT_ROOT}}", str(project_root))
    prompt = prompt.replace("{{BRAINSTORMER_REFERENCE_ROOT}}", str(brainstormer_reference_root))
    return (
        prompt
        + "\n\n"
        + "Cumulative paper context:\n"
        + f"- LaTeX: {_paper_tex_path(project_root)}\n"
        + f"- PDF: {_paper_pdf_path(project_root)}\n"
        + f"Executor research-validation file (must inspect): {iteration_dir / 'executor_research_validation.json'}\n"
        + "If `passed_min_research_gate` is false, still provide constructive recovery hypotheses and a concrete next-iteration plan.\n\n"
        + "Hard stop condition for this Codex run:\n"
        + "- Do one brainstorming pass for this iteration only.\n"
        + "- Write requested files, return summary, and terminate.\n"
        + "- Do not initiate additional autonomous cycles.\n"
    )


def _run_codex_exec(
    codex_bin: str,
    prompt_text: str,
    workdir: Path,
    output_last_message_file: Path,
    stdout_log_file: Path,
    stderr_log_file: Path,
    model: Optional[str],
    reasoning_effort: str,
    timeout_seconds: int,
) -> dict:
    cmd = [
        codex_bin,
        "exec",
        "--cd",
        str(workdir),
        "--skip-git-repo-check",
        "--dangerously-bypass-approvals-and-sandbox",
        "-c",
        f'model_reasoning_effort="{reasoning_effort}"',
        "-o",
        str(output_last_message_file),
        "-",
    ]
    if model:
        cmd.extend(["-m", model])

    started = time.time()
    with stdout_log_file.open("w", encoding="utf-8") as out_f, stderr_log_file.open(
        "w", encoding="utf-8"
    ) as err_f:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=out_f,
            stderr=err_f,
            text=True,
        )
        timed_out = False
        try:
            if timeout_seconds > 0:
                proc.communicate(input=prompt_text, timeout=timeout_seconds)
            else:
                proc.communicate(input=prompt_text)
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()
            proc.communicate()

    elapsed = time.time() - started
    return {
        "returncode": proc.returncode,
        "elapsed_seconds": elapsed,
        "cmd": cmd,
        "timed_out": timed_out,
        "timeout_seconds": timeout_seconds if timeout_seconds > 0 else None,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run autonomous executor/brainstormer loop with Codex CLI."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config.json",
    )
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--sleep-seconds", type=int, default=10)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--test-mode", action="store_true")
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default="low",
        choices=["low", "medium", "high", "xhigh"],
    )
    parser.add_argument(
        "--codex-timeout-seconds",
        type=int,
        default=0,
        help="Per-call timeout in seconds; set 0 to disable timeout.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = _load_loop_config(args.config.resolve(), args)

    runtime_dir = cfg.project_root / "runtime"
    iterations_dir = cfg.project_root / "iterations"
    reports_dir = cfg.project_root / "reports"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    iterations_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    stop_flag = runtime_dir / "STOP"
    status_file = runtime_dir / "loop_status.json"
    events_file = runtime_dir / "loop_events.jsonl"
    brainstormer_feedback_cache = runtime_dir / "latest_brainstormer_feedback.md"
    heartbeat_file = runtime_dir / "heartbeat.txt"

    executor_base_prompt = _read_text(cfg.executor_prompt_file)
    brainstormer_template = _read_text(cfg.brainstormer_prompt_template_file)

    # Resolve Codex CLI path once so detached runs do not depend on login-shell PATH.
    codex_path = cfg.codex_bin
    if "/" not in codex_path:
        resolved = shutil.which(codex_path)
        if resolved is None:
            raise FileNotFoundError(f"Could not resolve Codex CLI binary '{codex_path}'")
        codex_path = resolved
    else:
        if not Path(codex_path).exists():
            raise FileNotFoundError(f"Configured Codex CLI binary does not exist: {codex_path}")

    _append_jsonl(
        events_file,
        {
            "ts": _utc_now(),
            "event": "loop_start",
            "project_root": str(cfg.project_root),
            "max_iterations": cfg.max_iterations,
            "sleep_seconds": cfg.sleep_seconds,
            "model": cfg.model,
            "test_mode": cfg.test_mode,
            "reasoning_effort": cfg.reasoning_effort,
            "codex_timeout_seconds": cfg.codex_timeout_seconds,
        },
    )
    _write_json(
        status_file,
        {
            "ts": _utc_now(),
            "running": True,
            "completed_iterations": 0,
            "max_iterations": cfg.max_iterations,
            "sleep_seconds": cfg.sleep_seconds,
            "model": cfg.model,
            "test_mode": cfg.test_mode,
            "reasoning_effort": cfg.reasoning_effort,
            "codex_timeout_seconds": cfg.codex_timeout_seconds,
        },
    )

    completed = 0
    while True:
        _write_text(heartbeat_file, _utc_now())

        if stop_flag.exists():
            _append_jsonl(events_file, {"ts": _utc_now(), "event": "stop_flag_detected"})
            break

        if cfg.max_iterations is not None and completed >= cfg.max_iterations:
            _append_jsonl(events_file, {"ts": _utc_now(), "event": "max_iterations_reached", "completed": completed})
            break

        iteration_number = _next_iteration_number(iterations_dir)
        iter_dir = iterations_dir / f"iter_{iteration_number:04d}"
        iter_dir.mkdir(parents=True, exist_ok=False)
        _write_json(
            status_file,
            {
                "ts": _utc_now(),
                "running": True,
                "completed_iterations": completed,
                "current_iteration": iteration_number,
                "phase": "executor",
                "last_iteration_dir": str(iter_dir),
                "max_iterations": cfg.max_iterations,
                "sleep_seconds": cfg.sleep_seconds,
                "model": cfg.model,
                "test_mode": cfg.test_mode,
                "reasoning_effort": cfg.reasoning_effort,
                "codex_timeout_seconds": cfg.codex_timeout_seconds,
            },
        )

        brainstormer_feedback = ""
        if brainstormer_feedback_cache.exists():
            brainstormer_feedback = _read_text(brainstormer_feedback_cache)

        recent_hypothesis_outcomes = _recent_hypothesis_outcomes_text(
            iterations_dir=iterations_dir,
            current_iteration_number=iteration_number,
        )

        executor_prompt = _build_executor_prompt(
            base_prompt_text=executor_base_prompt,
            executor_prompt_file=cfg.executor_prompt_file,
            project_root=cfg.project_root,
            iteration_dir=iter_dir,
            iteration_number=iteration_number,
            brainstormer_feedback_text=brainstormer_feedback,
            recent_hypothesis_outcomes_text=recent_hypothesis_outcomes,
            test_mode=cfg.test_mode,
        )
        executor_prompt_file = iter_dir / "executor_prompt.md"
        _write_text(executor_prompt_file, executor_prompt)

        exec_last_message = iter_dir / "executor_last_message.md"
        exec_stdout = iter_dir / "executor_stdout.log"
        exec_stderr = iter_dir / "executor_stderr.log"

        _append_jsonl(events_file, {"ts": _utc_now(), "event": "executor_start", "iteration": iteration_number})
        try:
            exec_result = _run_codex_exec(
                codex_bin=codex_path,
                prompt_text=executor_prompt,
                workdir=cfg.executor_workdir,
                output_last_message_file=exec_last_message,
                stdout_log_file=exec_stdout,
                stderr_log_file=exec_stderr,
                model=cfg.model,
                reasoning_effort=cfg.reasoning_effort,
                timeout_seconds=cfg.codex_timeout_seconds,
            )
        except Exception as exc:
            _write_text(iter_dir / "executor_exception.txt", f"{type(exc).__name__}: {exc}\n")
            _append_jsonl(
                events_file,
                {
                    "ts": _utc_now(),
                    "event": "executor_exception",
                    "iteration": iteration_number,
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                },
            )
            break
        _append_jsonl(
            events_file,
            {
                "ts": _utc_now(),
                "event": "executor_finish",
                "iteration": iteration_number,
                **exec_result,
            },
        )
        executor_validation = _validate_executor_research_progress(
            iter_dir=iter_dir,
            project_root=cfg.project_root,
            iteration_number=iteration_number,
        )
        _write_json(iter_dir / "executor_research_validation.json", executor_validation)
        _append_jsonl(
            events_file,
            {
                "ts": _utc_now(),
                "event": "executor_validation",
                "iteration": iteration_number,
                **executor_validation,
            },
        )

        brainstormer_prompt = _build_brainstormer_prompt(
            brainstormer_template=brainstormer_template,
            project_root=cfg.project_root,
            iteration_dir=iter_dir,
            brainstormer_reference_root=cfg.brainstormer_reference_root,
        )
        brainstormer_prompt_file = iter_dir / "brainstormer_prompt.md"
        _write_text(brainstormer_prompt_file, brainstormer_prompt)

        rev_last_message = iter_dir / "brainstormer_last_message.md"
        rev_stdout = iter_dir / "brainstormer_stdout.log"
        rev_stderr = iter_dir / "brainstormer_stderr.log"

        _append_jsonl(events_file, {"ts": _utc_now(), "event": "brainstormer_start", "iteration": iteration_number})
        _write_json(
            status_file,
            {
                "ts": _utc_now(),
                "running": True,
                "completed_iterations": completed,
                "current_iteration": iteration_number,
                "phase": "brainstormer",
                "last_iteration_dir": str(iter_dir),
                "max_iterations": cfg.max_iterations,
                "sleep_seconds": cfg.sleep_seconds,
                "model": cfg.model,
                "test_mode": cfg.test_mode,
                "reasoning_effort": cfg.reasoning_effort,
                "codex_timeout_seconds": cfg.codex_timeout_seconds,
            },
        )
        try:
            rev_result = _run_codex_exec(
                codex_bin=codex_path,
                prompt_text=brainstormer_prompt,
                workdir=cfg.executor_workdir,
                output_last_message_file=rev_last_message,
                stdout_log_file=rev_stdout,
                stderr_log_file=rev_stderr,
                model=cfg.model,
                reasoning_effort=cfg.reasoning_effort,
                timeout_seconds=cfg.codex_timeout_seconds,
            )
        except Exception as exc:
            _write_text(iter_dir / "brainstormer_exception.txt", f"{type(exc).__name__}: {exc}\n")
            _append_jsonl(
                events_file,
                {
                    "ts": _utc_now(),
                    "event": "brainstormer_exception",
                    "iteration": iteration_number,
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                },
            )
            break
        _append_jsonl(
            events_file,
            {
                "ts": _utc_now(),
                "event": "brainstormer_finish",
                "iteration": iteration_number,
                **rev_result,
            },
        )

        latest_brainstormer_text = ""
        if rev_last_message.exists():
            latest_brainstormer_text = _read_text(rev_last_message)
            _write_text(brainstormer_feedback_cache, latest_brainstormer_text)

        iteration_meta = {
            "iteration_number": iteration_number,
            "timestamp_utc": _utc_now(),
            "executor": exec_result,
            "brainstormer": rev_result,
            "executor_last_message_file": str(exec_last_message),
            "brainstormer_last_message_file": str(rev_last_message),
        }
        _write_json(iter_dir / "iteration_meta.json", iteration_meta)

        completed += 1
        _write_json(
            status_file,
            {
                "ts": _utc_now(),
                "running": True,
                "completed_iterations": completed,
                "current_iteration": iteration_number,
                "phase": "sleep",
                "last_iteration_dir": str(iter_dir),
                "max_iterations": cfg.max_iterations,
                "sleep_seconds": cfg.sleep_seconds,
                "model": cfg.model,
                "test_mode": cfg.test_mode,
                "reasoning_effort": cfg.reasoning_effort,
                "codex_timeout_seconds": cfg.codex_timeout_seconds,
            },
        )

        if cfg.max_iterations is not None and completed >= cfg.max_iterations:
            continue

        time.sleep(max(cfg.sleep_seconds, 0))

    _write_json(
        status_file,
        {
            "ts": _utc_now(),
            "running": False,
            "completed_iterations": completed,
            "max_iterations": cfg.max_iterations,
            "model": cfg.model,
            "test_mode": cfg.test_mode,
            "reasoning_effort": cfg.reasoning_effort,
            "codex_timeout_seconds": cfg.codex_timeout_seconds,
        },
    )
    _append_jsonl(events_file, {"ts": _utc_now(), "event": "loop_exit", "completed_iterations": completed})


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - defensive guard for detached runs
        sys.stderr.write(f"[fatal] {exc}\n")
        raise
