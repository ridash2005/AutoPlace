from typing import Dict, List, Set, Tuple, Optional
import random
import numpy as np
from data import Cell, Net


def bisection_partition(cells: Dict[str, Cell], nets: List[Net], balance_tol: float, rng: random.Random) -> Tuple[Set[str], Set[str]]:
    cell_names = [c for c in cells if not cells[c].fixed]
    A, B = set(cell_names[::2]), set(cell_names[1::2])
    target = len(cell_names) / 2

    def cutsize(Aset, Bset):
        return sum(1 for n in nets if any(p in Aset for p in n.pins if p in cells) and any(p in Bset for p in n.pins if p in cells))

    best = cutsize(A, B)
    improved = True
    while improved:
        improved = False
        rng.shuffle(cell_names)
        for c in cell_names:
            if c in A and abs(len(A) - 1 - target) / target <= balance_tol:
                A.remove(c)
                B.add(c)
                new_cut = cutsize(A, B)
                if new_cut < best:
                    best = new_cut
                    improved = True
                else:
                    B.remove(c)
                    A.add(c)
            elif c in B and abs(len(B) - 1 - target) / target <= balance_tol:
                B.remove(c)
                A.add(c)
                new_cut = cutsize(A, B)
                if new_cut < best:
                    best = new_cut
                    improved = True
                else:
                    A.remove(c)
                    B.add(c)
    return A, B


def region_subtract_blockages(region: Tuple[float, float, float, float],
                              blockages: Optional[List[Tuple[float, float, float, float]]],
                              keepout: float) -> Optional[Tuple[float, float, float, float]]:
    x0, y0, x1, y1 = region
    if not blockages:
        return region
    for bx0, by0, bx1, by1 in blockages:
        bx0, by0, bx1, by1 = bx0 - keepout, by0 - keepout, bx1 + keepout, by1 + keepout
        if bx0 <= x0 and bx1 >= x1 and by0 <= y0 and by1 >= y1:
            return None
        if bx0 <= x1 and bx1 >= x0:
            if by0 <= y0 < by1 < y1:
                y0 = max(y0, by1)
            elif y0 < by0 < y1 <= by1:
                y1 = min(y1, by0)
        if by0 <= y1 and by1 >= y0:
            if bx0 <= x0 < bx1 < x1:
                x0 = max(x0, bx1)
            elif x0 < bx0 < x1 <= bx1:
                x1 = min(x1, bx0)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def recursive_bipartition_place(
    cells: Dict[str, Cell], nets: List[Net], chip_w: float, chip_h: float,
    num_parts: int, balance_tol: float, rng: random.Random,
    blockages: Optional[List[Tuple[float, float, float, float]]] = None, keepout: float = 0.0
) -> None:
    regions = [(0.0, 0.0, chip_w, chip_h, list(cells.keys()))]

    while len(regions) < num_parts:
        regions.sort(key=lambda r: len(r[4]), reverse=True)
        x0, y0, x1, y1, cnames = regions.pop(0)
        subcells = {n: cells[n] for n in cnames}
        A, B = bisection_partition(subcells, nets, balance_tol, rng)

        if blockages:
            adjusted = region_subtract_blockages((x0, y0, x1, y1), blockages, keepout)
            if adjusted is None:
                continue
            x0, y0, x1, y1 = adjusted

        w, h = x1 - x0, y1 - y0
        if w >= h:  # Horizontal split
            mid = x0 + w * len(A) / (len(A) + len(B) + 1e-9)
            regions += [(x0, y0, mid, y1, list(A)), (mid, y0, x1, y1, list(B))]
        else:  # Vertical split
            mid = y0 + h * len(A) / (len(A) + len(B) + 1e-9)
            regions += [(x0, y0, x1, mid, list(A)), (x0, mid, x1, y1, list(B))]

    for x0, y0, x1, y1, cnames in regions:
        for name in cnames:
            c = cells[name]
            if c.fixed:
                continue
            rx0, ry0, rx1, ry1 = x0, y0, x1 - c.w, y1 - c.h
            if c.region:
                r0, r1, r2, r3 = c.region
                rx0, ry0 = max(rx0, r0), max(ry0, r1)
                rx1, ry1 = min(rx1, r2 - c.w), min(ry1, r3 - c.h)
            if blockages:
                for bx0, by0, bx1, by1 in blockages:
                    bx0, by0, bx1, by1 = bx0 - keepout, by0 - keepout, bx1 + keepout, by1 + keepout
                    if rx0 < bx1 and rx1 > bx0 and ry0 < by1 and ry1 > by0:
                        if rx0 < bx1 < rx1: rx0 = bx1
                        if rx1 > bx0 > rx0: rx1 = bx0
                        if ry0 < by1 < ry1: ry0 = by1
                        if ry1 > by0 > ry0: ry1 = by0
            rx1 = max(rx1, rx0)
            ry1 = max(ry1, ry0)
            c.x = rng.uniform(rx0, rx1)
            c.y = rng.uniform(ry0, ry1)
