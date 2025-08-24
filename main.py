import random
from typing import Dict, Tuple, List, Optional
import numpy as np

from input import create_cpu_like_blockages, example_netlist_dict, CHIP_W, CHIP_H
from data import Cell, PlacementParams
from parser import parse_netlist, init_placement
from cost import total_wirelength, density_overflow, congestion_estimate
from global_placement import quadratic_global_placement
from partitioning import recursive_bipartition_place
from annealing import anneal
from visualize import plot_cells_and_nets, plot_congestion


def run_placement(
    netlist: Dict = example_netlist_dict,
    chip_w: float = CHIP_W,
    chip_h: float = CHIP_H,
    init_coords: Optional[Dict[str, Tuple[float, float]]] = None,
    num_partitions: int = 4,
    balance_tolerance: float = 0.1,
    blockages: Optional[List[Tuple[float, float, float, float]]] = None,
    keepout: float = 0.0,
    anneal_iters: int = 5000,
    rng_seed: int = 42,
    visualize: bool = True,
):
    rng = random.Random(rng_seed)
    cells, nets = parse_netlist(netlist)
    if blockages is None:
        blockages = create_cpu_like_blockages(cells, rng_seed=rng_seed)

    params = PlacementParams(
        chip_w=chip_w, chip_h=chip_h, num_partitions=num_partitions,
        balance_tolerance=balance_tolerance, blockages=blockages,
        keepout=keepout, anneal_iters=anneal_iters, rng_seed=rng_seed,
        visualize=visualize
    )

    init_placement(cells, chip_w, chip_h, init_coords, rng)

    print("Starting placement flow...")
    print("1. Quadratic global placement...")
    quadratic_global_placement(cells, nets, chip_w, chip_h, rng)

    print("2. Recursive bisection partitioning...")
    recursive_bipartition_place(cells, nets, chip_w, chip_h, num_partitions, balance_tolerance, rng)

    print("3. Simulated annealing refinement...")
    final_cost, acc_ratio = anneal(nets, cells, params, rng)

    print("\nPlacement complete. Calculating metrics...")
    wl = total_wirelength(nets, cells)
    dens_over, util = density_overflow(cells, chip_w, chip_h)
    cong = congestion_estimate(nets, cells, chip_w, chip_h)

    placed = {c.name: (c.x, c.y) for c in cells.values()}
    metrics = {
        "wirelength_total": wl,
        "wirelength_per_net": wl / max(1, len(nets)),
        "density_overflow_sum": dens_over,
        "avg_utilization": float(np.mean(util)),
        "congestion_heatmap": cong,
        "anneal_accept_ratio": acc_ratio,
        "final_cost": final_cost,
    }

    print("Final Wirelength:", wl)
    print("Density Overflow:", dens_over)
    print("Simulated Annealing Acceptance Ratio:", acc_ratio)

    if visualize:
        plot_cells_and_nets(cells, nets, params, "Final Placement")
        plot_congestion(cong, params, "Final Congestion")

    return placed, metrics


if __name__ == "__main__":
    placed_cells, placement_metrics = run_placement(anneal_iters=25000, visualize=True)
