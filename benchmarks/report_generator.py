"""
benchmarks/report_generator.py
═══════════════════════════════
Step 10 — Benchmark Report Generator

Collects all JSON results produced by the other bench_*.py scripts and
generates a unified Markdown + JSON report suitable for saving to the repo
or sharing with the team.

Inputs  (any subset is fine — missing files are skipped gracefully)
------
  results/fps.json           from bench_fps.py
  results/multi_session.json from bench_multi_session.py
  results/accuracy.json      from bench_rep_accuracy.py
  results/memory.json        from bench_memory.json
  results/inference.json     from tools/benchmark_inference.py  (Step 8)

Outputs
-------
  results/BENCHMARK_REPORT.md    Human-readable summary
  results/BENCHMARK_REPORT.json  Machine-readable full report

Usage
-----
  # Run all benchmarks first, then generate:
  python benchmarks/report_generator.py

  # Custom input directory:
  python benchmarks/report_generator.py --results-dir my_results/

  # Custom output:
  python benchmarks/report_generator.py --output reports/v8_report
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("report_gen")


# ── Argument parsing ───────────────────────────────────────────────────────────

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate benchmark report generator")
    p.add_argument("--results-dir", default="results", help="Directory containing bench JSON files")
    p.add_argument("--output",      default="results/BENCHMARK_REPORT",
                   help="Output base path (no extension — .md and .json added)")
    return p.parse_args()


# ── JSON loader ────────────────────────────────────────────────────────────────

def _load(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        log.warning("Could not load %s: %s", path, e)
        return None


# ── System info ────────────────────────────────────────────────────────────────

def _system_info() -> Dict:
    info = {
        "python":   platform.python_version(),
        "platform": platform.platform(),
        "cpu":      platform.processor() or platform.machine(),
    }
    try:
        import torch
        info["pytorch"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu"]        = torch.cuda.get_device_name(0)
            info["cuda"]       = torch.version.cuda
            info["vram_gb"]    = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    except ImportError:
        pass
    try:
        import tensorrt as trt
        info["tensorrt"] = trt.__version__
    except ImportError:
        info["tensorrt"] = "not installed"
    try:
        import onnxruntime as ort
        info["onnxruntime"]   = ort.__version__
        info["ort_providers"] = ort.get_available_providers()
    except ImportError:
        info["onnxruntime"] = "not installed"
    return info


# ── Section renderers ──────────────────────────────────────────────────────────

def _render_fps(data: Dict) -> str:
    cfg = data.get("config", {})
    tl  = data.get("total_latency", {})
    fps = data.get("effective_fps", 0)
    lines = [
        "### FPS & Per-Stage Latency (Single Session)\n",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Effective FPS | **{fps}** |",
        f"| Total p50 latency | {tl.get('p50', '?')} ms |",
        f"| Total p90 latency | {tl.get('p90', '?')} ms |",
        f"| Total p99 latency | {tl.get('p99', '?')} ms |",
        f"| Exercise | {cfg.get('exercise', '?')} |",
        f"| BiLSTM | {cfg.get('bilstm', '?')} |",
        "",
        "**Per-stage breakdown:**\n",
        "| Stage | p50 ms | p90 ms | p99 ms | min | max |",
        "|-------|--------|--------|--------|-----|-----|",
    ]
    for stage, s in data.get("stages", {}).items():
        if s:
            lines.append(
                f"| {stage} | {s.get('p50','?')} | {s.get('p90','?')} | "
                f"{s.get('p99','?')} | {s.get('min','?')} | {s.get('max','?')} |"
            )
    return "\n".join(lines)


def _render_multi(data: Dict) -> str:
    agg = data.get("aggregate", {})
    cfg = data.get("config", {})
    lines = [
        "### Multi-Session Concurrent Stress Test\n",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Sessions | {cfg.get('n_sessions', '?')} |",
        f"| FPS target / session | {cfg.get('fps', '?')} |",
        f"| Test duration | {cfg.get('duration_s', '?')} s |",
        f"| Total frames sent | {agg.get('total_frames_sent', '?')} |",
        f"| Delivery rate | **{agg.get('delivery_rate_pct', '?')}%** |",
        f"| Backpressure events | {agg.get('total_backpressure', '?')} |",
        f"| Total throughput | {agg.get('throughput_total_fps', '?')} fps |",
        f"| Per-session throughput | {agg.get('throughput_per_session_fps', '?')} fps |",
        f"| Latency p50 | {agg.get('p50_ms', '?')} ms |",
        f"| Latency p90 | {agg.get('p90_ms', '?')} ms |",
        f"| Latency p99 | {agg.get('p99_ms', '?')} ms |",
        f"| Avg tracker stability | {round(agg.get('avg_track_stability', 0) * 100, 1)}% |",
        f"| Total ID switches | {agg.get('total_id_switches', '?')} |",
    ]

    # Ramp results (if present)
    if "ramp_results" in data:
        lines += [
            "",
            "**Ramp results:**\n",
            "| Sessions | fps/session | p50 ms | p90 ms | Delivery % |",
            "|----------|-------------|--------|--------|------------|",
        ]
        for r in data["ramp_results"]:
            a = r.get("aggregate", {})
            lines.append(
                f"| {r['config']['n_sessions']} | {a.get('throughput_per_session_fps','?')} | "
                f"{a.get('p50_ms','?')} | {a.get('p90_ms','?')} | {a.get('delivery_rate_pct','?')}% |"
            )

    return "\n".join(lines)


def _render_accuracy(data: Dict) -> str:
    summ = data.get("summary", {})
    by_ex = data.get("by_exercise", {})
    lines = [
        "### Rep Counting Accuracy\n",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total tests | {summ.get('total_tests', '?')} |",
        f"| Passed | {summ.get('passed', '?')} |",
        f"| Failed | {summ.get('failed', '?')} |",
        f"| Average accuracy | **{round(summ.get('avg_accuracy', 0) * 100, 1)}%** |",
        f"| Overall result | **{summ.get('overall', '?')}** |",
        "",
        "**Pass rate by exercise:**\n",
        "| Exercise | Pass rate |",
        "|----------|-----------|",
    ]
    for ex, rate in sorted(by_ex.items(), key=lambda x: x[1]):
        icon = "✓" if rate == 100.0 else ("⚠" if rate >= 80 else "✗")
        lines.append(f"| {ex} | {icon} {rate:.0f}% |")

    # Failed tests
    failed = [t for t in data.get("tests", []) if not t.get("passed")]
    if failed:
        lines += ["", "**Failed tests:**\n",
                  "| Exercise | Test | Expected | Actual | Accuracy |",
                  "|----------|------|----------|--------|----------|"]
        for t in failed:
            lines.append(
                f"| {t['exercise']} | {t['test']} | {t['expected']} | "
                f"{t['actual']} | {round(t['accuracy']*100,0):.0f}% |"
            )

    return "\n".join(lines)


def _render_memory(data: Dict) -> str:
    ram = data.get("ram", {})
    gpu = data.get("gpu", {})
    perf = data.get("performance", {})
    lines = [
        "### Memory Stability\n",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| RAM baseline | {ram.get('baseline_mb', '?')} MB |",
        f"| RAM peak | {ram.get('peak_mb', '?')} MB |",
        f"| RAM final | {ram.get('final_mb', '?')} MB |",
        f"| RAM growth / 1k frames | {ram.get('growth_per_1k_mb', '?')} MB |",
        f"| RAM leak detected | {'⚠ YES' if ram.get('potential_leak') else '✓ No'} |",
    ]
    if gpu.get("monitored") is not False:
        lines += [
            f"| GPU memory peak | {gpu.get('peak_mb', '?')} MB |",
            f"| GPU growth / 1k frames | {gpu.get('growth_per_1k_mb', '?')} MB |",
            f"| GPU leak detected | {'⚠ YES' if gpu.get('potential_leak') else '✓ No'} |",
        ]
    lines += [
        f"| Effective FPS (memory run) | {perf.get('effective_fps', '?')} |",
        f"| Verdict | **{data.get('verdict', '?')}** |",
    ]
    if data.get("issues"):
        lines.append("\n**Issues:**\n")
        for issue in data["issues"]:
            lines.append(f"- ⚠ {issue}")
    return "\n".join(lines)


def _render_inference(data: Dict) -> str:
    lines = [
        "### Inference Backend Comparison (Step 8)\n",
        "| Backend | Component | p50 ms | p90 ms | FPS |",
        "|---------|-----------|--------|--------|-----|",
    ]
    for r in data.get("results", []):
        if r.get("error"):
            lines.append(f"| {r['backend']} | {r['component']} | ERROR | {r['error'][:30]} | - |")
        else:
            lines.append(
                f"| {r['backend']} | {r['component']} | {r.get('p50_ms','?')} | "
                f"{r.get('p90_ms','?')} | {r.get('fps','?')} |"
            )
    sys_info = data.get("system", {})
    if sys_info.get("gpu"):
        lines.append(f"\n*GPU: {sys_info['gpu']}  CUDA: {sys_info.get('cuda','?')}  TRT: {sys_info.get('tensorrt_version','?')}*")
    return "\n".join(lines)


# ── Markdown document builder ──────────────────────────────────────────────────

def _build_markdown(
    sys_info: Dict,
    fps_data: Optional[Dict],
    multi_data: Optional[Dict],
    acc_data: Optional[Dict],
    mem_data: Optional[Dict],
    inf_data: Optional[Dict],
    generated_at: str,
) -> str:
    sections = [
        f"# AI Fitness Trainer — Benchmark Report",
        f"",
        f"**Generated:** {generated_at}  ",
        f"**System:** {sys_info.get('gpu', 'CPU')} · CUDA {sys_info.get('cuda', 'N/A')} · "
        f"PyTorch {sys_info.get('pytorch', 'N/A')} · Python {sys_info.get('python', 'N/A')}",
        f"",
        f"---",
        f"",
    ]

    if fps_data:
        sections.append(_render_fps(fps_data))
        sections.append("\n---\n")

    if multi_data:
        sections.append(_render_multi(multi_data))
        sections.append("\n---\n")

    if acc_data:
        sections.append(_render_accuracy(acc_data))
        sections.append("\n---\n")

    if mem_data:
        sections.append(_render_memory(mem_data))
        sections.append("\n---\n")

    if inf_data:
        sections.append(_render_inference(inf_data))
        sections.append("\n---\n")

    sections.append("*Report generated by `benchmarks/report_generator.py`*")
    return "\n".join(sections)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args  = _parse()
    rd    = Path(args.results_dir)
    out_base = Path(args.output)
    out_base.parent.mkdir(parents=True, exist_ok=True)

    log.info("═" * 60)
    log.info("  Benchmark Report Generator")
    log.info("  results dir: %s", rd)
    log.info("═" * 60)

    fps_data   = _load(rd / "fps.json")
    multi_data = _load(rd / "multi_session.json")
    acc_data   = _load(rd / "accuracy.json")
    mem_data   = _load(rd / "memory.json")
    inf_data   = _load(rd / "inference.json")

    loaded = sum(1 for d in [fps_data, multi_data, acc_data, mem_data, inf_data] if d)
    log.info("  Loaded %d / 5 benchmark result files", loaded)

    if loaded == 0:
        log.warning("No benchmark results found in %s. Run bench_*.py scripts first.", rd)
        log.warning("  Expected files: fps.json, multi_session.json, accuracy.json, memory.json, inference.json")
        sys.exit(0)

    sys_info     = _system_info()
    generated_at = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    # ── Markdown ──────────────────────────────────────────────────────────────
    md = _build_markdown(sys_info, fps_data, multi_data, acc_data, mem_data, inf_data, generated_at)
    md_path = Path(str(out_base) + ".md")
    md_path.write_text(md)
    log.info("  Markdown → %s", md_path)

    # ── JSON ──────────────────────────────────────────────────────────────────
    full_report = {
        "generated_at": generated_at,
        "system":       sys_info,
        "fps":          fps_data,
        "multi_session": multi_data,
        "rep_accuracy": acc_data,
        "memory":       mem_data,
        "inference":    inf_data,
    }
    json_path = Path(str(out_base) + ".json")
    with open(json_path, "w") as f:
        json.dump(full_report, f, indent=2)
    log.info("  JSON     → %s", json_path)

    log.info("═" * 60)
    log.info("  Done.")


if __name__ == "__main__":
    main()
