"""
Interactive Chat with LatticeNN - Sentence Generation
Uses generate_sequence() for word-by-word generation with:
- Trigram history scoring (H)
- Spring stiffness scoring (S)
- Activation scoring (D)
- Context decay scoring (C)
- Role-based grammar scoring (phi)
- Causal directional boost

Usage:
    python3 -m lrn.chat_session [--level=second_grade] [--max-tokens=15]
"""
import sys
import os
import json
import time
from datetime import datetime

sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn import LatticeNN, propagate, _is_stable, generate, generate_sequence
from lrn.generate import _score_candidates
from lrn.sensory_grounding import sensory_grounding
from lrn.harmonic_lessons import harmonic_video_training
from lrn.physical_manipulation import physical_manipulation
from lrn.social_interaction import social_interaction
from lrn.trainer import train
from lrn.corpora import get_corpus, get_corpus_info


CHAT_DIR = "/Users/tyarc/github/lrn/reports/chat"
os.makedirs(CHAT_DIR, exist_ok=True)


def build_lattice(level="second_grade", verbose=False):
    """Build a trained lattice up to the specified level."""
    print(f"\n  Building lattice up to {level}...")
    t0 = time.time()

    lnn = sensory_grounding(verbose=False)

    physical_manipulation(lnn, verbose=False)
    social_interaction(lnn, verbose=False)

    levels = ["prek", "kindergarten", "first_grade", "second_grade", "third_grade"]
    target_idx = levels.index(level) if level in levels else 4

    for lvl in levels[:target_idx + 1]:
        if verbose:
            print(f"  Training {lvl}...")
        harmonic_video_training(lnn, lvl, verbose=False)
        corpus = get_corpus(lvl)
        learn_type = "sensory" if lvl == "prek" else "language"
        train(lnn, corpus, reps=50, learn_type=learn_type, rem_interval="end", verbose=False)

    elapsed = time.time() - t0
    print(f"  Lattice ready: {len(lnn.nodes)} nodes, {len(lnn.springs)} springs ({elapsed:.1f}s)")
    return lnn


def show_generation_trace(lnn, context, top_k=5):
    """Show candidate scoring for the next word."""
    candidates = _score_candidates(lnn, context, top_k=top_k)
    if not candidates:
        return None

    print(f"\n    Next word candidates (context: {' '.join(context[-3:])}):")
    for i, c in enumerate(candidates):
        bar = "#" * max(1, c["score"] // 200)
        print(f"    {i+1}. {c['word']:20s} score={c['score']:6d} {bar}  act={c['activation']} role={c['role']}")

    return candidates[0]["word"]


def generate_with_trace(lnn, prompt, max_tokens=15, top_k=5, show_trace=True):
    """Generate a sequence word by word, optionally showing candidates."""
    context = prompt.lower().split()
    generated = []

    for step in range(max_tokens):
        candidates = generate(lnn, context, top_k=top_k)
        if not candidates:
            break

        if show_trace:
            best = show_generation_trace(lnn, context, top_k=top_k)
        else:
            best = candidates[0]["word"]

        if best is None:
            break

        generated.append(best)
        context.append(best)

        node = lnn.nodes.get(best)
        if node:
            node.activation = max(node.activation, 50)

        propagate(lnn, n_steps=2, verbose=False)

    return " ".join(generated)


def chat_session(lnn, max_tokens=15, max_turns=20):
    """Interactive chat session with sentence generation."""
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_file = os.path.join(CHAT_DIR, f"chat_{session_id}.json")

    session_data = {
        "session": session_id,
        "lattice_state": {
            "nodes": len(lnn.nodes),
            "springs": len(lnn.springs),
        },
        "turns": [],
    }

    print("\n" + "=" * 60)
    print("  LRN Chat - Sentence Generation")
    print("=" * 60)
    print("  Type your message. LRN generates responses word by word.")
    print("  Commands: 'trace on/off' (show candidates), 'quit' to exit")
    print("=" * 60)

    show_trace = True

    for turn in range(max_turns):
        try:
            user_input = input("\n  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if user_input.lower() in ("quit", "exit", "q"):
            break

        if not user_input:
            continue

        if user_input.lower() == "trace on":
            show_trace = True
            print("  [Trace enabled]")
            continue
        elif user_input.lower() == "trace off":
            show_trace = False
            print("  [Trace disabled]")
            continue

        print(f"\n  LRN: ", end="", flush=True)
        t0 = time.time()
        response = generate_with_trace(lnn, user_input, max_tokens=max_tokens, show_trace=show_trace)
        elapsed = time.time() - t0

        if response:
            print(f"\n  -> {response}")
        else:
            print("\n  -> [no response generated]")

        print(f"  ({elapsed:.1f}s, {len(response.split()) if response else 0} words)")

        session_data["turns"].append({
            "human": user_input,
            "lattice": response,
            "time": round(elapsed, 2),
        })

    with open(session_file, "w") as f:
        json.dump(session_data, f, indent=2)

    print(f"\n  Session saved: {session_file}")
    print(f"  Turns: {len(session_data['turns'])}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Chat with LatticeNN")
    parser.add_argument("--level", default="second_grade", help="Curriculum level to train")
    parser.add_argument("--max-tokens", type=int, default=15, help="Max tokens per response")
    parser.add_argument("--max-turns", type=int, default=20, help="Max conversation turns")
    parser.add_argument("--quick", action="store_true", help="Skip trace output")
    args = parser.parse_args()

    lnn = build_lattice(level=args.level, verbose=True)
    chat_session(lnn, max_tokens=args.max_tokens, max_turns=args.max_turns)
