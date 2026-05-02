"""
Run full sequential English training with optimized REM synthesis.
"""
import sys
import time

sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn.english_training import SequentialTrainer


def main():
    print("=" * 60)
    print("FULL SEQUENTIAL ENGLISH TRAINING")
    print("=" * 60)
    
    start_time = time.time()
    
    trainer = SequentialTrainer(max_reps_per_stage=100, max_stall_rounds=2)
    results = trainer.train_all_stages()
    
    elapsed = time.time() - start_time
    
    # Summary
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE - {elapsed:.1f}s")
    print(f"{'='*60}")
    
    summary = trainer.get_summary()
    print(f"\nFinal: {summary['total_nodes']} nodes, {summary['total_springs']} springs")
    
    gates = summary["gates"]
    print(f"Gates: {gates['passed']}/{gates['total']} passed ({gates['percentage']}%)")
    
    for stage, result in summary["stages"].items():
        status = "PASS" if result["gate_passed"] else "FAIL"
        print(f"  {stage}: {status} ({result['gate_metric']}, {result['total_reps']} reps)")
    
    return summary


if __name__ == "__main__":
    main()
