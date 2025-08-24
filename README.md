# mini_vlsi_placement_tool

This project implements a comprehensive VLSI placement optimization flow.

It aims to compute optimized cell locations on a chip to minimize total wirelength, reduce routing congestion, and satisfy density and blockage constraints.

The placement flow consists of:  

Quadratic Global Placement: Computes initial cell locations by minimizing squared wirelength through solving linear systems. 

Recursive Bisection Partitioning: Divides chip floorplan into balanced, non-overlapping regions using partitioning heuristics and blockage-aware constraints.

Simulated Annealing Refinement: Applies local perturbations via metaheuristic optimization to improve quality based on multi-objective cost metrics.

Visualization: Calculates wirelength, density overflow, congestion heatmaps, and blockage penalty with easy-to-interpret visual output.
