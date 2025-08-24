from typing import Dict, List
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import cg
import random
from data import Cell, Net

def quadratic_global_placement(
    cells: Dict[str, Cell], 
    nets: List[Net], 
    chip_w: float, 
    chip_h: float, 
    rng: random.Random
) -> None:
    n = len(cells)
    idx_map = {name: i for i, name in enumerate(cells)}
    L = lil_matrix((n, n))

    for net in nets:
        pins = [idx_map[p] for p in net.pins if p in idx_map]
        for i in pins:
            L[i, i] += len(pins) - 1
            for j in pins:
                if i != j:
                    L[i, j] -= 1

    x0 = np.array([c.x for c in cells.values()])
    y0 = np.array([c.y for c in cells.values()])

    x_sol, _ = cg(L.tocsr(), np.zeros(n), x0=x0)
    y_sol, _ = cg(L.tocsr(), np.zeros(n), x0=y0)

    for name, i in idx_map.items():
        c = cells[name]
        if not c.fixed:
            c.x = np.clip(x_sol[i], 0, chip_w - c.w)
            c.y = np.clip(y_sol[i], 0, chip_h - c.h)
