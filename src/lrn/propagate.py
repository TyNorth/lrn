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


def propagate_with_negative(lnn: LatticeNN, n_steps: int = 1, verbose: bool = False):
    """Enhanced propagation that properly handles negative stiffness (arithmetic)."""
    for step in range(n_steps):
        new_activations = {}
        
        for name, node in lnn.nodes.items():
            if node.pinned:
                new_activations[name] = node.activation
                continue

            neighbors = lnn.get_neighbors(name)
            if not neighbors:
                new_activations[name] = node.activation
                continue

            positive_sum = 0
            negative_sum = 0
            pos_total = 0
            neg_total = 0

            for neighbor_name, sp in neighbors:
                neighbor_node = lnn.nodes.get(neighbor_name)
                if not neighbor_node or neighbor_node.activation == 0:
                    continue

                tau_w = TAU_W[sp.tau]
                eff_k = sp.stiffness * tau_w
                neighbor_act = neighbor_node.activation

                if eff_k > 0:
                    positive_sum += eff_k * neighbor_act
                    pos_total += eff_k
                elif eff_k < 0:
                    negative_sum += abs(eff_k) * neighbor_act
                    neg_total += abs(eff_k)

            self_retention = node.activation * 4

            if pos_total > 0:
                neighbor_pos = (positive_sum * 6) // pos_total
            else:
                neighbor_pos = 0

            if neg_total > 0:
                neighbor_neg = (negative_sum * 6) // neg_total
            else:
                neighbor_neg = 0

            net_influence = neighbor_pos - neighbor_neg
            new_act = (self_retention + net_influence) // 10
            new_activations[name] = max(0, min(100, new_act))

        for name, act in new_activations.items():
            lnn.nodes[name].activation = act

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