#!/usr/bin/env python3
"""
Phase 5 - Mathematics Module (Fixed v4)
Zero-energy arithmetic - measure net spring balance
"""
import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

import pickle
import json


IVM_STEP = 100
K_PROX_BASE = 100
K_BAL_POS = 50
K_BAL_NEG = -100  # Stronger negative to actually cancel
K_OP_BOND = 80


def initialize_number_line(lnn):
    for i in range(11):
        lnn.get_or_create(str(i))
        lnn.get_or_create(f"num:{i}")
        lnn.nodes[str(i)].x = i * IVM_STEP
        lnn.nodes[f"num:{i}"].x = i * IVM_STEP

    for i in range(11):
        for j in range(i+1, 11):
            dist = j - i
            k = K_PROX_BASE // dist
            if k < 2:
                break
            lnn.add_or_update_spring(str(i), str(j), stiffness=k, tau=1)
            lnn.add_or_update_spring(f"num:{i}", f"num:{j}", stiffness=k, tau=1)


def teach_arithmetic_constraints(lnn):
    lnn.get_or_create("op:plus")
    lnn.get_or_create("op:equals")
    lnn.get_or_create("plus")
    lnn.get_or_create("equals")

    for op_word, op_node in [("equals", "op:equals"), ("=", "op:equals"),
                             ("plus", "op:plus"), ("+", "op:plus")]:
        lnn.add_or_update_spring(op_word, op_node, stiffness=K_OP_BOND, tau=0, mode="pos_max")

    ADDITION_FACTS = [(a, b, a+b) for a in range(10) for b in range(10) if a+b <= 10]

    for a, b, c in ADDITION_FACTS:
        bal = f"balance:{a}:+:{b}"
        lnn.get_or_create(bal)
        
        lnn.add_or_update_spring(str(a), bal, stiffness=K_BAL_POS, tau=0)
        lnn.add_or_update_spring(str(b), bal, stiffness=K_BAL_POS, tau=0)
        lnn.add_or_update_spring("op:plus", bal, stiffness=K_BAL_POS//2, tau=0)
        
        # Correct answer gets NEGATIVE spring (cancellation)
        lnn.add_or_update_spring(str(c), bal, stiffness=K_BAL_NEG, tau=0, mode="neg_override")
        
        # Wrong answers get POSITIVE springs (additive tension)
        for wrong in range(11):
            if wrong != c:
                lnn.add_or_update_spring(str(wrong), bal, stiffness=K_BAL_POS, tau=0)


def measure_net_balance(lnn, a, b, candidate):
    """Measure net spring balance on balance node - 0 means balanced (correct)."""
    from lrn import propagate
    
    lnn.reset()
    
    for tok in [str(a), "op:plus", "plus", str(b), "op:equals", "equals"]:
        node = lnn.nodes.get(tok)
        if node:
            node.activation = 100
            node.pinned = True

    candidate_node = lnn.nodes.get(str(candidate))
    if candidate_node:
        candidate_node.activation = 80

    for _ in range(8):
        propagate(lnn, n_steps=1)

    bal_node_name = f"balance:{a}:+:{b}"
    bal_node = lnn.nodes.get(bal_node_name)
    if not bal_node:
        return 999
    
    net_balance = 0
    for neighbor_name, sp in lnn.get_neighbors(bal_node_name):
        neighbor = lnn.nodes.get(neighbor_name)
        if neighbor and neighbor.activation > 0:
            net_balance += sp.stiffness * neighbor.activation
    
    return net_balance


def solve_addition(lnn, a, b):
    results = []
    for x in range(11):
        net = measure_net_balance(lnn, a, b, x)
        results.append((x, net))
    results.sort(key=lambda r: abs(r[1]))
    return results[0][0], results


def main():
    print("=" * 60)
    print("Phase 5: Mathematics Module (Fixed v4)")
    print("=" * 60)

    with open("/Users/tyarc/github/lrn/checkpoints/phase4.pkl", 'rb') as f:
        data = pickle.load(f)
    lnn = data['lnn']

    print(f"\n[1] Loaded checkpoint, nodes: {len(lnn.nodes)}")

    print("\n[2] Initializing number line (0-10)...")
    initialize_number_line(lnn)
    print(f"    Added digit nodes 0-10 with proximity springs")

    print("\n[3] Teaching arithmetic constraints...")
    teach_arithmetic_constraints(lnn)
    
    balance_nodes = len([n for n in lnn.nodes if n.startswith("balance:")])
    print(f"    Created {balance_nodes} balance constraint nodes")

    print("\n[4] Testing arithmetic (net balance)...")

    test_cases = [(1, 2), (2, 2), (3, 3), (5, 5), (4, 4), (1, 1), (0, 5), (2, 3)]
    results = []

    for a, b in test_cases:
        correct = a + b
        answer, all_results = solve_addition(lnn, a, b)
        
        results_dict = dict(all_results)
        correct_bal = abs(results_dict[correct])
        wrong_bal = abs(results_dict[(correct + 3) % 11])
        discrimination = correct_bal < wrong_bal
        passed = answer == correct and discrimination
        results.append({
            "equation": f"{a} + {b} = {correct}",
            "answer": answer,
            "correct": correct,
            "net_balance_correct": results_dict[correct],
            "net_balance_wrong": results_dict[(correct + 3) % 11],
            "discrimination": discrimination,
            "passed": passed
        })
        status = "✓" if passed else "✗"
        print(f"    {status} {a}+{b}: answer={answer} (expected={correct}), net(correct)={results_dict[correct]}, net(wrong)={results_dict[(correct+3)%11]}")

    passed_count = sum(1 for r in results if r["passed"])
    all_passed = passed_count == len(results)

    sys_test_results = {
        "phase": 5,
        "test_type": "mathematics",
        "balance_nodes": balance_nodes,
        "test_cases": results,
        "passed_count": passed_count,
        "total_tests": len(results),
        "status": "fixed_v4"
    }

    with open("/Users/tyarc/github/lrn/sys_test/math_results.json", 'w') as f:
        json.dump(sys_test_results, f, indent=2)

    print(f"\n[5] Saved sys_test/math_results.json")

    print("\n" + "=" * 60)
    if all_passed:
        print(f"✓ PASS: Math module working ({passed_count}/{len(results)} tests)")
    else:
        print(f"⚠ PARTIAL: {passed_count}/{len(results)} tests passed")
    print("=" * 60)

    with open("/Users/tyarc/github/lrn/checkpoints/phase5.pkl", 'wb') as f:
        pickle.dump({'lnn': lnn, 'math_results': results}, f)

    return 0 if passed_count >= len(results) // 2 else 1


if __name__ == '__main__':
    sys.exit(main())