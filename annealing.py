import math
import random
import numpy as np
from typing import Dict, List, Tuple, Optional
from data import Cell, Net, PlacementParams
from cost import cost_multiobj_weighted


def legalize_and_clip(c: Cell, chip_w: float, chip_h: float,
                      blockages: Optional[List[Tuple[float, float, float, float]]] = None,
                      keepout: float = 0.0) -> None:
    """Ensure cell stays within chip bounds and outside blockages."""
    c.x = np.clip(c.x, 0, chip_w - c.w)
    c.y = np.clip(c.y, 0, chip_h - c.h)

    if not blockages:
        return

    for x0, y0, x1, y1 in blockages:
        bx0, by0, bx1, by1 = x0 - keepout, y0 - keepout, x1 + keepout, y1 + keepout
        cx0, cy0, cx1, cy1 = c.x, c.y, c.x + c.w, c.y + c.h

        if cx1 > bx0 and cx0 < bx1 and cy1 > by0 and cy0 < by1:
            overlap_x = min(cx1, bx1) - max(cx0, bx0)
            overlap_y = min(cy1, by1) - max(cy0, by0)

            if overlap_x < overlap_y:  # Resolve along x
                c.x = np.clip(c.x - overlap_x if cx0 < bx0 else c.x + overlap_x, 0, chip_w - c.w)
            else:  # Resolve along y
                c.y = np.clip(c.y - overlap_y if cy0 < by0 else c.y + overlap_y, 0, chip_h - c.h)


def anneal(nets: List[Net], cells: Dict[str, Cell], params: PlacementParams, rng: random.Random) -> Tuple[float, float]:
    """
    Simulated annealing local refinement minimizing multi-objective cost.
    """
    T0, T_end = 1.0, 1e-3
    iters = max(1000, params.anneal_iters)
    cost = cost_multiobj_weighted(nets, cells, params)
    movable = [c for c in cells.values() if not c.fixed]
    T, accept = T0, 0

    for it in range(iters):
        if not movable:
            break
        c = rng.choice(movable)
        oldx, oldy = c.x, c.y
        step = max(params.chip_w, params.chip_h) * 0.02

        c.x += rng.gauss(0, step)
        c.y += rng.gauss(0, step)

        legalize_and_clip(c, params.chip_w, params.chip_h, params.blockages, params.keepout)

        if c.region:
            r0, r1, r2, r3 = c.region
            c.x = np.clip(c.x, r0, r2 - c.w)
            c.y = np.clip(c.y, r1, r3 - c.h)

        new_cost = cost_multiobj_weighted(nets, cells, params)
        d = new_cost - cost

        if d <= 0 or rng.random() < math.exp(-d / max(T, 1e-9)):
            cost = new_cost
            accept += 1
        else:
            c.x, c.y = oldx, oldy

        T = T0 * ((T_end / T0) ** (it / iters))

    return cost, accept / iters
