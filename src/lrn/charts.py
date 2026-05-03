"""
ASCII Chart Rendering for LRN Reports
"""

def bar_chart(label, score, total, width=40):
    pct = score / max(1, total)
    filled = int(pct * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"  {label:20s} |{bar}| {score:5.1f}/{total} ({pct*100:5.1f}%)"


def spring_bar(label, stiffness, max_stiffness, width=40):
    pct = stiffness / max(1, max_stiffness)
    filled = int(pct * width)
    bar = "█" * filled
    return f"  {label:15s} |{bar} ({stiffness})"


def tau_distribution(tau_counts):
    tau_names = {
        0: "τ=0 Constitutive",
        1: "τ=1 Definitional",
        2: "τ=2 Causal",
        3: "τ=3 Categorical",
        4: "τ=4 Contextual",
    }
    max_count = max(tau_counts.values()) if tau_counts else 1
    lines = []
    lines.append(f"  {'='*62}")
    lines.append(f"  SPRING DISTRIBUTION (Tau Hierarchy)")
    lines.append(f"  {'='*62}")
    for tau in range(5):
        count = tau_counts.get(tau, 0)
        bar_len = int((count / max_count) * 40) if max_count > 0 else 0
        bar = "█" * bar_len
        lines.append(f"  {tau_names[tau]:20s} |{bar} {count}")
    return "\n".join(lines)


def category_density(categories_data):
    lines = []
    lines.append(f"  {'='*62}")
    lines.append(f"  CATEGORY CLUSTERING DENSITY")
    lines.append(f"  {'='*62}")
    for cat_name, tau3, total in categories_data:
        density = tau3 / max(1, total)
        bar_len = int(density * 40)
        bar = "█" * bar_len + "░" * (40 - bar_len)
        status = "DENSE" if density >= 0.5 else "SPARSE"
        lines.append(f"  {cat_name:15s} |{bar}| {tau3}/{total} ({density*100:.0f}%) {status}")
    return "\n".join(lines)


def probe_result(query, connections, width=40):
    if not connections:
        return f"  No connections found for '{query}'"
    
    max_k = max(c.get("stiffness", 0) for c in connections)
    lines = []
    lines.append(f"  {'='*62}")
    lines.append(f"  PROBE: '{query}'")
    lines.append(f"  {'='*62}")
    for conn in connections:
        name = conn.get("name", "?")
        k = conn.get("stiffness", 0)
        tau = conn.get("tau", 4)
        bar_len = int((k / max(1, max_k)) * width)
        bar = "█" * bar_len
        lines.append(f"  {name:15s} τ={tau} k={k:4d} |{bar}")
    return "\n".join(lines)


def compare_result(a, b, shared, only_a, only_b, width=30):
    lines = []
    lines.append(f"  {'='*62}")
    lines.append(f"  COMPARE: '{a}' vs '{b}'")
    lines.append(f"  {'='*62}")
    
    if shared:
        max_k = max(c.get("stiffness", 0) for c in shared)
        lines.append(f"\n  Shared associations:")
        for conn in shared:
            name = conn.get("name", "?")
            k = conn.get("stiffness", 0)
            tau = conn.get("tau", 4)
            bar_len = int((k / max(1, max_k)) * width)
            bar = "█" * bar_len
            lines.append(f"    {name:15s} τ={tau} k={k:4d} |{bar}")
    
    if only_a:
        max_k = max(c.get("stiffness", 0) for c in only_a)
        lines.append(f"\n  Only in '{a}':")
        for conn in only_a:
            name = conn.get("name", "?")
            k = conn.get("stiffness", 0)
            tau = conn.get("tau", 4)
            bar_len = int((k / max(1, max_k)) * width)
            bar = "█" * bar_len
            lines.append(f"    {name:15s} τ={tau} k={k:4d} |{bar}")
    
    if only_b:
        max_k = max(c.get("stiffness", 0) for c in only_b)
        lines.append(f"\n  Only in '{b}':")
        for conn in only_b:
            name = conn.get("name", "?")
            k = conn.get("stiffness", 0)
            tau = conn.get("tau", 4)
            bar_len = int((k / max(1, max_k)) * width)
            bar = "█" * bar_len
            lines.append(f"    {name:15s} τ={tau} k={k:4d} |{bar}")
    
    return "\n".join(lines)


def confidence_label(confidence):
    if confidence >= 0.7:
        return ""
    elif confidence >= 0.3:
        return " [CONFIDENCE: MEDIUM]"
    else:
        return " [CONFIDENCE: LOW]"


def progress_bar(current, total, width=40):
    pct = current / max(1, total)
    filled = int(pct * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"\r  [{bar}] {current}/{total} ({pct*100:.0f}%)"


def chat_turn(human, lattice, confidence=None, width=60):
    lines = []
    lines.append(f"  You: {human}")
    conf = confidence_label(confidence) if confidence else ""
    lines.append(f"  LRN: {lattice}{conf}")
    return "\n".join(lines)
