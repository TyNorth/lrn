import sys
sys.path.insert(0, '/Users/tyarc/github/lrn/src')

from lrn.lattice import LatticeNN, TAU_W


def propagate(lnn: LatticeNN, n_steps: int = 1, verbose: bool = False):
    for step in range(n_steps):
        for name, node in lnn.nodes.items():
            if node.pinned:
                continue

            neighbors = lnn.get_neighbors(name)
            if not neighbors:
                continue

            weighted_sum = 0
            stiff_total = 0

            for neighbor_name, sp in neighbors:
                neighbor_node = lnn.nodes.get(neighbor_name)
                if not neighbor_node:
                    continue

                tau_w = TAU_W[sp.tau]
                eff_k = sp.stiffness * tau_w

                if eff_k == 0:
                    continue

                weighted_sum += eff_k * neighbor_node.activation
                stiff_total += abs(eff_k)

            if stiff_total > 0:
                neighbor_influence = (weighted_sum * 6) // stiff_total
            else:
                neighbor_influence = 0

            new_act = (node.activation * 4 + neighbor_influence) // 10
            node.activation = max(0, min(100, new_act))

        if verbose:
            active = sum(1 for n in lnn.nodes.values() if n.activation > 0)
            print(f"Step {step + 1}: {active} active nodes")


def _is_stable(lnn: LatticeNN, threshold: int = 1) -> bool:
    changes = 0
    for name, node in lnn.nodes.items():
        if node.pinned:
            continue
        neighbors = lnn.get_neighbors(name)
        if not neighbors:
            continue

        weighted_sum = 0
        stiff_total = 0

        for neighbor_name, sp in neighbors:
            neighbor_node = lnn.nodes.get(neighbor_name)
            if not neighbor_node:
                continue

            tau_w = TAU_W[sp.tau]
            eff_k = sp.stiffness * tau_w
            if eff_k == 0:
                continue

            weighted_sum += eff_k * neighbor_node.activation
            stiff_total += abs(eff_k)

        if stiff_total > 0:
            neighbor_influence = (weighted_sum * 6) // stiff_total
        else:
            neighbor_influence = 0

        new_act = (node.activation * 4 + neighbor_influence) // 10
        new_act = max(0, min(100, new_act))

        if abs(new_act - node.activation) > threshold:
            changes += 1

    return changes == 0