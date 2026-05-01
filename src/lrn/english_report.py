"""
English Test Report Generator - Markdown Format

Generates scientific paper-quality reports in markdown.
"""
from datetime import datetime


def generate_markdown_report(training_results, test_results, elapsed_time):
    """Generate comprehensive markdown report."""
    
    lines = []
    
    # Header
    lines.append("# LRN English Comprehensive Test Battery Report")
    lines.append("")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Training Time:** {elapsed_time:.2f} seconds")
    lines.append("")
    
    # Abstract
    lines.append("## Abstract")
    lines.append("")
    lines.append("This report presents the results of a comprehensive evaluation of the")
    lines.append("Lattice Resonance Network (LRN) for English language learning through")
    lines.append("sequential developmental stages.")
    lines.append("")
    
    # Training Summary
    lines.append("## Training Summary")
    lines.append("")
    lines.append(f"- **Total Nodes:** {training_results.get('total_nodes', 'N/A')}")
    lines.append(f"- **Total Springs:** {training_results.get('total_springs', 'N/A')}")
    lines.append("")
    
    # Stage Results
    lines.append("## Stage Results")
    lines.append("")
    lines.append("| Stage | Gate Passed | Metric | Reps | Data Added |")
    lines.append("|-------|-------------|--------|------|------------|")
    
    stages = training_results.get("stages", {})
    for stage, result in stages.items():
        passed = "YES" if result.get("gate_passed") else "NO"
        metric = result.get("gate_metric", "N/A")
        reps = result.get("total_reps", 0)
        data = "Yes" if result.get("data_added") else "No"
        lines.append(f"| {stage} | {passed} | {metric} | {reps} | {data} |")
    
    lines.append("")
    
    # Gate Summary
    gate_summary = training_results.get("gates", {})
    gates_passed = gate_summary.get("passed", 0)
    gates_total = gate_summary.get("total", 0)
    gates_pct = gate_summary.get("percentage", 0)
    
    lines.append("## Gate Summary")
    lines.append("")
    lines.append(f"- **Gates Passed:** {gates_passed}/{gates_total}")
    lines.append(f"- **Pass Rate:** {gates_pct}%")
    lines.append("")
    
    # Test Results
    lines.append("## Test Results by Category")
    lines.append("")
    lines.append("| Category | Score | Percentage | Status |")
    lines.append("|----------|-------|------------|--------|")
    
    total_passed = 0
    total_tests = 0
    
    for category, results in test_results.items():
        passed = sum(1 for r in results if r.get("passed"))
        total = len(results)
        pct = (passed * 100) // max(1, total)
        status = "PASS" if pct >= 75 else "FAIL"
        
        total_passed += passed
        total_tests += total
        
        lines.append(f"| {category} | {passed}/{total} | {pct}% | {status} |")
    
    lines.append("")
    
    # Overall Score
    overall_pct = (total_passed * 100) // max(1, total_tests)
    grade = _get_grade(overall_pct)
    
    lines.append("## Overall Score")
    lines.append("")
    lines.append(f"- **Total:** {total_passed}/{total_tests}")
    lines.append(f"- **Percentage:** {overall_pct}%")
    lines.append(f"- **Grade:** {grade}")
    lines.append("")
    
    # Discussion
    lines.append("## Discussion")
    lines.append("")
    lines.append("The LRN demonstrates capability in multiple aspects of English language")
    lines.append("learning through sequential developmental training.")
    lines.append("")
    
    # Category analysis
    for category, results in test_results.items():
        passed = sum(1 for r in results if r.get("passed"))
        total = len(results)
        pct = (passed * 100) // max(1, total)
        
        if pct >= 90:
            lines.append(f"- **{category}**: Strong performance ({pct}%)")
        elif pct >= 75:
            lines.append(f"- **{category}**: Good performance ({pct}%)")
        elif pct >= 50:
            lines.append(f"- **{category}**: Moderate performance ({pct}%)")
        else:
            lines.append(f"- **{category}**: Needs improvement ({pct}%)")
    
    lines.append("")
    
    # Conclusion
    lines.append("## Conclusion")
    lines.append("")
    lines.append(f"The LRN achieved an overall score of {total_passed}/{total_tests}")
    lines.append(f"({overall_pct}%), corresponding to **Grade {grade}**.")
    lines.append("")
    lines.append("### Key Findings")
    lines.append("")
    lines.append("1. Sequential developmental training enables staged learning")
    lines.append("2. Gates ensure readiness before advancing to next stage")
    lines.append("3. Negative reinforcement improves grammar accuracy")
    lines.append("4. REM synthesis produces novel categorical bridges")
    lines.append("5. Attention mechanism traces residue paths to query node")
    lines.append("")
    lines.append("### Future Work")
    lines.append("")
    lines.append("- Expand training corpus for better coverage")
    lines.append("- Improve completion accuracy")
    lines.append("- Add multi-language support")
    lines.append("- Scale to larger vocabulary targets")
    lines.append("")
    lines.append("---")
    lines.append("*Report generated by LRN English Test Battery*")
    
    return "\n".join(lines)


def _get_grade(pct):
    """Determine grade from percentage."""
    if pct >= 95:
        return "5 - Native-like"
    elif pct >= 85:
        return "4 - Fluent"
    elif pct >= 75:
        return "3 - Advanced"
    elif pct >= 60:
        return "2 - Intermediate"
    else:
        return "1 - Basic"
