"""
Interactive Chat with LRN - Confidence based on spring stiffness
"""
import sys
import os
import json
from datetime import datetime
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn import propagate
from lrn.natural_tokenize import learn_from_text
from lrn.inference import add_word_nodes
from lrn.trainer import optimal_rem


CHAT_DIR = "/Users/tyarc/github/lrn/reports/chat"
os.makedirs(CHAT_DIR, exist_ok=True)


def get_confidence(lnn, query):
    """Confidence = average spring stiffness from query node / max possible."""
    query_node = f"word:{query}"
    if query_node not in lnn.nodes:
        return 0.0
    
    neighbors = lnn.get_neighbors(query_node)
    if not neighbors:
        return 0.0
    
    avg_stiffness = sum(sp.stiffness for _, sp in neighbors) / len(neighbors)
    # Normalize: max reasonable stiffness is ~1000
    return min(1.0, avg_stiffness / 1000.0)


def generate_response(lnn, query, propagate_steps=3):
    """Generate response by activating query and collecting top activated nodes."""
    # Ingest query into lattice
    learn_from_text(lnn, query, repetitions=1, learn_type="language")
    add_word_nodes(lnn, [query])
    
    # Reset and pin query words
    words = query.lower().split()
    for n in lnn.nodes.values():
        n.activation = 0
    
    for word in words:
        query_node = f"word:{word}"
        if query_node in lnn.nodes:
            lnn.nodes[query_node].activation = 100
    
    propagate(lnn, n_steps=propagate_steps)
    
    # Collect activated word nodes
    activated = []
    for name, node in lnn.nodes.items():
        if name.startswith("word:") and node.activation > 5:
            word = name.replace("word:", "")
            if word not in words:  # Don't repeat query words
                activated.append((word, node.activation))
    
    activated.sort(key=lambda x: x[1], reverse=True)
    
    # Build response from top activated words
    response_words = [w for w, _ in activated[:8]]
    
    if not response_words:
        return "I don't know about that yet.", 0.0
    
    # Calculate confidence from spring stiffness
    confidence = get_confidence(lnn, words[0] if words else "")
    
    # Form natural response
    response = " ".join(response_words[:5])
    
    return response, confidence


def chat_session(lnn, max_turns=20):
    """Interactive chat session."""
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
    
    wake_buffer = []
    print("\n  LRN Chat — type 'quit' to exit")
    print("  " + "=" * 50)
    
    for turn in range(max_turns):
        try:
            user_input = input("\n  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        
        if user_input.lower() in ("quit", "exit", "q"):
            break
        
        if not user_input:
            continue
        
        response, confidence = generate_response(lnn, user_input)
        
        # Add uncertainty flag
        if confidence >= 0.7:
            display = response
        elif confidence >= 0.3:
            display = f"{response} [CONFIDENCE: MEDIUM]"
        else:
            display = f"{response} [CONFIDENCE: LOW]"
        
        print(f"  LRN: {display}")
        
        # Track for REM
        wake_buffer.append(user_input)
        wake_buffer.append(response)
        if len(wake_buffer) > 20:
            wake_buffer = wake_buffer[-20:]
        
        # Periodic REM during chat
        if (turn + 1) % 5 == 0:
            optimal_rem(lnn, wake_buffer)
            propagate(lnn, n_steps=3)
        
        # Save turn
        session_data["turns"].append({
            "human": user_input,
            "lattice": response,
            "confidence": round(confidence, 3),
            "activated_nodes": [n for n in lnn.nodes if n.startswith("word:") and lnn.nodes[n].activation > 5],
        })
    
    # Save session
    session_data["lattice_state"]["nodes"] = len(lnn.nodes)
    session_data["lattice_state"]["springs"] = len(lnn.springs)
    
    with open(session_file, "w") as f:
        json.dump(session_data, f, indent=2)
    
    print(f"\n  Session saved: {session_file}")
    print(f"  Turns: {len(session_data['turns'])}")
