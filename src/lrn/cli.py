"""
LRN CLI - Lattice Relaxation Network Command Line Interface

Usage:
    lrn_train train <level> [--reps=N] [--rem-interval=N]
    lrn_train ingest <file> [--level=prek] [--rem-interval=N]
    lrn_train test <level>
    lrn_train probe <word> [--depth=N]
    lrn_train probe --category <name>
    lrn_train probe --compare <word1> <word2>
    lrn_train chat
    lrn_train report <level>
    lrn_train compare <file1> <file2>
    lrn_train levels
    lrn_train status
"""
import sys
import os
import json
import argparse
from datetime import datetime

sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn.trainer import create_lnn, train
from lrn.assessor import assess_level
from lrn.sensory_grounding import sensory_grounding
from lrn.harmonic_lessons import harmonic_video_training
from lrn.charts import (
    bar_chart, tau_distribution, category_density,
    probe_result, compare_result, chat_turn, confidence_label,
    progress_bar
)
from lrn.corpora import get_corpus, get_corpus_info, AVAILABLE_LEVELS
from lrn.ingest import ingest_book
from lrn.probe import probe, probe_category, compare
from lrn.chat import chat_session


REPORT_DIR = "/Users/tyarc/github/lrn/reports"


def cmd_train(args):
    """Train a level."""
    level = args.level
    reps = args.reps
    rem_interval = args.rem_interval
    if rem_interval == "end":
        rem_label = "end only"
    elif int(rem_interval) == 1:
        rem_label = "every sentence"
    else:
        rem_label = f"every {rem_interval} sentences"
        rem_interval = int(rem_interval)
    
    print(f"\n{'='*60}")
    print(f"TRAINING: {level.upper()}")
    print(f"{'='*60}")
    
    corpus_info = get_corpus_info(level)
    print(f"  Original: {corpus_info['original']} sentences")
    print(f"  Varied: {corpus_info['varied']} sentences")
    print(f"  Combined: {corpus_info['total']} sentences")
    print(f"  Reps: {reps}")
    print(f"  REM interval: {rem_label}")
    print()
    
    # Step 1 & 2: Sensory grounding (interoceptive + external)
    lnn = sensory_grounding(verbose=True)
    
    # Step 3: Harmonic video labeling (always present, content varies by level)
    harmonic_video_training(lnn, level, verbose=True)
    
    # Step 4: Text corpus training
    import time
    t0 = time.time()
    corpus = get_corpus(level)
    train(lnn, corpus, reps=reps, learn_type="sensory" if level == "prek" else "language", 
          rem_interval=rem_interval)
    train_time = time.time() - t0
    
    print(f"\n  Training: {train_time:.1f}s")
    print(f"  Nodes: {len(lnn.nodes)}, Springs: {len(lnn.springs)}")
    
    # Assess
    print(f"\n{'='*60}")
    print(f"ASSESSMENT: {level.upper()}")
    print(f"{'='*60}")
    
    results = assess_level(lnn, level)
    
    # Group by domain
    domains = {}
    for name, r in results.items():
        domain = r.get("domain", "General")
        if domain not in domains:
            domains[domain] = []
        domains[domain].append((name, r))
    
    for domain, skills in domains.items():
        print(f"\n  ── {domain} ──")
        for name, r in skills:
            skill_name = r.get("skill", name.upper())
            print(bar_chart(f"  {skill_name}", r["score"], r["possible"]))
            # Show failing/partial items
            if "items" in r:
                fails = [(item, status, detail) for item, status, detail in r["items"] if status == "FAIL"]
                partials = [(item, status, detail) for item, status, detail in r["items"] if status == "PARTIAL"]
                if partials:
                    for item, _, detail in partials[:3]:
                        print(f"    ~ {item}: {detail}")
                if fails:
                    for item, _, detail in fails[:3]:
                        print(f"    ✗ {item}: {detail}")
    
    total_score = sum(r["score"] for r in results.values())
    total_possible = sum(r["possible"] for r in results.values())
    mastery_pct = (total_score * 100) // total_possible
    
    print(f"\n  {'─'*62}")
    print(bar_chart("TOTAL", total_score, total_possible))
    
    status = "MASTERY ✓" if mastery_pct >= 100 else "IN PROGRESS"
    print(f"\n  Status: {status} ({mastery_pct}%)")
    
    # Tau distribution
    tau_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    for sp in lnn.springs.values():
        tau_counts[sp.tau] += 1
    print(tau_distribution(tau_counts))
    
    # Category density
    categories_data = []
    for cat_name, members in [("animals", ["cat", "dog", "bird", "fish", "horse", "cow", "sheep", "pig", "chicken", "duck"]),
                               ("colors", ["red", "blue", "green", "yellow", "orange", "purple", "black", "white", "brown", "pink"]),
                               ("emotions", ["happy", "sad", "angry", "scared", "surprised", "tired"])]:
        tau3 = 0
        total = 0
        for i in range(len(members)):
            for j in range(i+1, len(members)):
                key = lnn._key(f"word:{members[i]}", f"word:{members[j]}")
                total += 1
                if key in lnn.springs and lnn.springs[key].tau == 3:
                    tau3 += 1
        categories_data.append((cat_name, tau3, total))
    print(category_density(categories_data))
    
    # Save report
    level_dir = os.path.join(REPORT_DIR, level)
    os.makedirs(level_dir, exist_ok=True)
    
    report = {
        "level": level,
        "date": datetime.now().isoformat(),
        "training_time": train_time,
        "reps": reps,
        "rem_interval": rem_interval,
        "nodes": len(lnn.nodes),
        "springs": len(lnn.springs),
        "tau_distribution": tau_counts,
        "total_score": total_score,
        "total_possible": total_possible,
        "mastery_pct": mastery_pct,
        "mastery_status": status,
        "assessments": {
            name: {
                "score": r["score"],
                "possible": r["possible"],
                "pct": r["pct"],
                "domain": r.get("domain", "General"),
                "skill": r.get("skill", name),
                "items": r.get("items", []),
            }
            for name, r in results.items()
        },
    }
    
    report_path = os.path.join(level_dir, f"{level}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n  Report saved: {report_path}")


def cmd_ingest(args):
    """Ingest a book."""
    filepath = args.file
    level = args.level
    rem_interval = args.rem_interval
    
    if not os.path.exists(filepath):
        print(f"  Error: File not found: {filepath}")
        return
    
    print(f"\n{'='*60}")
    print(f"INGESTING: {filepath}")
    print(f"{'='*60}")
    
    lnn = create_lnn()
    result = ingest_book(lnn, filepath, level=level, rem_interval=rem_interval)
    
    print(f"  Nodes: {result['nodes']}, Springs: {result['springs']}")
    
    # Save report
    ingest_dir = os.path.join(REPORT_DIR, "ingested")
    os.makedirs(ingest_dir, exist_ok=True)
    
    report = {
        "type": "ingested",
        "date": datetime.now().isoformat(),
        "file": filepath,
        "metadata": result["metadata"],
        "paragraphs": result["paragraphs"],
        "rem_cycles": result["rem_cycles"],
        "nodes": result["nodes"],
        "springs": result["springs"],
    }
    
    report_path = os.path.join(ingest_dir, f"{os.path.splitext(os.path.basename(filepath))[0]}_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"  Report saved: {report_path}")


def cmd_test(args):
    """Test a level without training."""
    level = args.level
    print(f"\n{'='*60}")
    print(f"TESTING: {level.upper()} (no training)")
    print(f"{'='*60}")
    print(f"  Note: This tests a fresh lattice. Use 'train' first for meaningful results.")
    
    lnn = create_lnn()
    results = assess_level(lnn, level)
    
    for name, r in results.items():
        print(bar_chart(name.upper(), r["score"], r["possible"]))
    
    total_score = sum(r["score"] for r in results.values())
    total_possible = sum(r["possible"] for r in results.values())
    mastery_pct = (total_score * 100) // total_possible
    print(f"  {'─'*62}")
    print(bar_chart("TOTAL", total_score, total_possible))


def cmd_probe(args):
    """Probe the lattice."""
    if args.category:
        result = probe_category(create_lnn(), args.category)
        if result:
            print(f"\n  Category: {result['category']}")
            print(f"  Members: {', '.join(result['members'])}")
            print(f"  τ=3 bridges: {result['tau3']}/{result['total']} ({result['density']*100:.0f}%)")
        else:
            print(f"  Unknown category: {args.category}")
        return
    
    if args.compare:
        if len(args.compare) != 2:
            print("  Usage: lrn_train probe --compare <word1> <word2>")
            return
        result = compare(create_lnn(), args.compare[0], args.compare[1])
        if result:
            print(compare_result(result["a"], result["b"], result["shared"], result["only_a"], result["only_b"]))
        else:
            print(f"  Could not compare '{args.compare[0]}' vs '{args.compare[1]}'")
        return
    
    if args.word:
        connections = probe(create_lnn(), args.word, depth=args.depth)
        if connections:
            print(probe_result(args.word, connections))
        else:
            print(f"  No connections found for '{args.word}'")
        return
    
    print("  Usage: lrn_train probe <word> [--depth=N]")
    print("         lrn_train probe --category <name>")
    print("         lrn_train probe --compare <word1> <word2>")


def cmd_chat(args):
    """Interactive chat."""
    lnn = create_lnn()
    chat_session(lnn)


def cmd_report(args):
    """Show last report with charts."""
    level = args.level
    level_dir = os.path.join(REPORT_DIR, level)
    
    if not os.path.exists(level_dir):
        print(f"  No reports found for level: {level}")
        return
    
    reports = sorted([f for f in os.listdir(level_dir) if f.endswith(".json")], reverse=True)
    if not reports:
        print(f"  No reports found for level: {level}")
        return
    
    report_path = os.path.join(level_dir, reports[0])
    with open(report_path, "r") as f:
        report = json.load(f)
    
    print(f"\n{'='*60}")
    print(f"REPORT: {report['level'].upper()} ({report['date']})")
    print(f"{'='*60}")
    
    if "assessments" in report:
        for name, r in report["assessments"].items():
            print(bar_chart(name.upper(), r["score"], r["possible"]))
    
    print(f"  {'─'*62}")
    print(bar_chart("TOTAL", report["total_score"], report["total_possible"]))
    print(f"\n  Status: {report['mastery_status']} ({report['mastery_pct']}%)")
    
    if "tau_distribution" in report:
        print(tau_distribution(report["tau_distribution"]))


def cmd_compare(args):
    """Compare two reports."""
    files = args.files
    if len(files) != 2:
        print("  Usage: lrn_train compare <file1> <file2>")
        return
    
    reports = []
    for f in files:
        with open(f, "r") as fh:
            reports.append(json.load(fh))
    
    print(f"\n{'='*60}")
    print(f"COMPARING: {reports[0]['level']} vs {reports[1]['level']}")
    print(f"{'='*60}")
    
    r1, r2 = reports[0], reports[1]
    
    print(f"\n  {'Metric':20s} {'Report 1':>12s} {'Report 2':>12s}")
    print(f"  {'─'*46}")
    print(f"  {'Nodes':20s} {r1['nodes']:>12d} {r2['nodes']:>12d}")
    print(f"  {'Springs':20s} {r1['springs']:>12d} {r2['springs']:>12d}")
    print(f"  {'Score':20s} {r1['total_score']:>8.1f}/{r1['total_possible']:>2d} {r2['total_score']:>8.1f}/{r2['total_possible']:>2d}")
    print(f"  {'Mastery %':20s} {r1['mastery_pct']:>10.1f}% {r2['mastery_pct']:>10.1f}%")


def cmd_levels(args):
    """List available levels."""
    print(f"\n{'='*60}")
    print(f"AVAILABLE LEVELS")
    print(f"{'='*60}")
    
    for level in AVAILABLE_LEVELS:
        info = get_corpus_info(level)
        level_dir = os.path.join(REPORT_DIR, level)
        report_count = 0
        last_report = "none"
        if os.path.exists(level_dir):
            reports = [f for f in os.listdir(level_dir) if f.endswith(".json")]
            report_count = len(reports)
            if reports:
                last_report = sorted(reports)[-1]
        
        print(f"\n  {level.upper()}")
        print(f"    Corpus: {info['total']} sentences ({info['original']} original + {info['varied']} varied)")
        print(f"    Reports: {report_count}")
        print(f"    Last: {last_report}")


def cmd_status(args):
    """Show current lattice state."""
    print(f"\n{'='*60}")
    print(f"LRN STATUS")
    print(f"{'='*60}")
    print(f"  Available levels: {', '.join(AVAILABLE_LEVELS)}")
    
    for level in AVAILABLE_LEVELS:
        info = get_corpus_info(level)
        print(f"  {level}: {info['total']} sentences")
    
    print(f"\n  Commands: train, ingest, test, probe, chat, report, compare, levels, status")


def main():
    parser = argparse.ArgumentParser(
        description="LRN - Lattice Relaxation Network CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # train
    p_train = subparsers.add_parser("train", help="Train a level")
    p_train.add_argument("level", choices=AVAILABLE_LEVELS, help="Level to train")
    p_train.add_argument("--reps", type=int, default=50, help="Training repetitions")
    p_train.add_argument("--rem-interval", default=1, help="REM every N sentences (1=every, end=final only)")
    p_train.set_defaults(func=cmd_train)
    
    # ingest
    p_ingest = subparsers.add_parser("ingest", help="Ingest a book")
    p_ingest.add_argument("file", help="Path to book text file")
    p_ingest.add_argument("--level", default="prek", help="Training level")
    p_ingest.add_argument("--rem-interval", type=int, default=50, help="REM every N paragraphs")
    p_ingest.set_defaults(func=cmd_ingest)
    
    # test
    p_test = subparsers.add_parser("test", help="Test a level")
    p_test.add_argument("level", choices=AVAILABLE_LEVELS, help="Level to test")
    p_test.set_defaults(func=cmd_test)
    
    # probe
    p_probe = subparsers.add_parser("probe", help="Probe the lattice")
    p_probe.add_argument("word", nargs="?", help="Word to probe")
    p_probe.add_argument("--depth", type=int, default=1, help="Propagation depth")
    p_probe.add_argument("--category", help="Test category clustering")
    p_probe.add_argument("--compare", nargs=2, metavar=("WORD1", "WORD2"), help="Compare two words")
    p_probe.set_defaults(func=cmd_probe)
    
    # chat
    p_chat = subparsers.add_parser("chat", help="Interactive chat")
    p_chat.set_defaults(func=cmd_chat)
    
    # report
    p_report = subparsers.add_parser("report", help="Show last report")
    p_report.add_argument("level", choices=AVAILABLE_LEVELS, help="Level to report")
    p_report.set_defaults(func=cmd_report)
    
    # compare
    p_compare = subparsers.add_parser("compare", help="Compare two reports")
    p_compare.add_argument("files", nargs=2, help="Two report JSON files")
    p_compare.set_defaults(func=cmd_compare)
    
    # levels
    p_levels = subparsers.add_parser("levels", help="List available levels")
    p_levels.set_defaults(func=cmd_levels)
    
    # status
    p_status = subparsers.add_parser("status", help="Show current state")
    p_status.set_defaults(func=cmd_status)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    args.func(args)


if __name__ == "__main__":
    main()
