from typing import Optional, Dict, List, Tuple
import numpy as np
from data import Cell, Net, PlacementParams


def wirelength_for_net(net: Net, cells: Dict[str, Cell]) -> float:
    coords = [(cells[p].x + c.w / 2, cells[p].y + c.h / 2) for p in net.pins if (c := cells.get(p))]
    if not coords:
        return 0.0
    xs, ys = zip(*coords)
    return (max(xs) - min(xs)) + (max(ys) - min(ys))


def total_wirelength(nets: List[Net], cells: Dict[str, Cell]) -> float:
    return sum(wirelength_for_net(n, cells) for n in nets)


def density_overflow(cells: Dict[str, Cell], chip_w: float, chip_h: float, bins: int = 16) -> Tuple[float, np.ndarray]:
    bin_w, bin_h = chip_w / bins, chip_h / bins
    grid = np.zeros((bins, bins))
    for c in cells.values():
        ix, iy = min(bins - 1, int((c.x + c.w / 2) // bin_w)), min(bins - 1, int((c.y + c.h / 2) // bin_h))
        grid[iy, ix] += c.w * c.h
    util = grid / (bin_w * bin_h)
    overflow = np.maximum(util - 1.0, 0.0)
    return float(np.sum(overflow)), util


def congestion_estimate(nets: List[Net], cells: Dict[str, Cell], chip_w: float, chip_h: float, bins: int = 32) -> np.ndarray:
    bin_w, bin_h = chip_w / bins, chip_h / bins
    heat = np.zeros((bins, bins))
    for n in nets:
        coords = [(cells[p].x + c.w / 2, cells[p].y + c.h / 2) for p in n.pins if (c := cells.get(p))]
        if not coords:
            continue
        xs, ys = zip(*coords)
        ix0, ix1 = max(0, int(min(xs) // bin_w)), min(bins - 1, int(max(xs) // bin_w))
        iy0, iy1 = max(0, int(min(ys) // bin_h)), min(bins - 1, int(max(ys) // bin_h))
        heat[iy0:iy1 + 1, ix0:ix1 + 1] += 1
    return heat


def blockage_penalty(c: Cell, blockages: Optional[List[Tuple[float, float, float, float]]], keepout: float) -> float:
    if not blockages:
        return 0.0
    cx0, cy0, cx1, cy1 = c.x, c.y, c.x + c.w, c.y + c.h
    pen = 0.0
    for x0, y0, x1, y1 in blockages:
        bx0, by0, bx1, by1 = x0 - keepout, y0 - keepout, x1 + keepout, y1 + keepout
        ox0, oy0 = max(cx0, bx0), max(cy0, by0)
        ox1, oy1 = min(cx1, bx1), min(cy1, by1)
        if ox1 > ox0 and oy1 > oy0:
            pen += (ox1 - ox0) * (oy1 - oy0)
    return pen


def clustering_penalty(cells: Dict[str, Cell]) -> float:
    movable = [c for c in cells.values() if not c.fixed]
    pen = 0.0
    for i, c1 in enumerate(movable):
        for c2 in movable[i + 1:]:
            dist = ((c1.x - c2.x) ** 2 + (c1.y - c2.y) ** 2) ** 0.5
            pen += 1.0 / (1.0 + dist)
    return pen


def density_overflow_weighted(cells: Dict[str, Cell], chip_w: float, chip_h: float, bins: int = 16) -> Tuple[float, np.ndarray]:
    bin_w, bin_h = chip_w / bins, chip_h / bins
    center_x, center_y = chip_w / 2, chip_h / 2
    grid = np.zeros((bins, bins))
    for c in cells.values():
        ix = min(bins - 1, int((c.x + c.w / 2) // bin_w))
        iy = min(bins - 1, int((c.y + c.h / 2) // bin_h))
        grid[iy, ix] += c.w * c.h
    util = grid / (bin_w * bin_h)
    overflow = np.maximum(util - 1.0, 0.0)

    weights = np.zeros((bins, bins))
    for iy in range(bins):
        for ix in range(bins):
            dx = max(abs((ix + 0.5) * bin_w - center_x), bin_w * 0.1)
            dy = max(abs((iy + 0.5) * bin_h - center_y), bin_h * 0.1)
            weights[iy, ix] = 1 / dx + 1 / dy
    weights /= np.mean(weights)

    return float(np.sum(overflow * weights)), util


def cost_multiobj_weighted(nets: List[Net], cells: Dict[str, Cell], params: PlacementParams) -> float:
    wl = total_wirelength(nets, cells)
    dens_over, _ = density_overflow_weighted(cells, params.chip_w, params.chip_h)
    pen = sum(blockage_penalty(c, params.blockages, params.keepout) for c in cells.values())
    return wl + 1* dens_over + 1* pen + 100000* clustering_penalty(cells)
