import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, List, Optional
from data import Cell, Net, PlacementParams


def plot_cells_and_nets(
    cells: Dict[str, Cell], nets: List[Net], params: PlacementParams,
    title: str = "VLSI Placement", cell_to_region: Optional[Dict[str, int]] = None
) -> None:
    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    ax.add_patch(patches.Rectangle((0, 0), params.chip_w, params.chip_h, fill=False, linewidth=1.5))

    if params.blockages:
        for x0, y0, x1, y1 in params.blockages:
            ax.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0, alpha=0.2, color='gray'))

    for net in nets:
        coords = [(cells[p].x + cells[p].w / 2, cells[p].y + cells[p].h / 2) for p in net.pins if p in cells]
        if len(coords) >= 2:
            xs, ys = zip(*coords)
            ax.plot([min(xs), max(xs), max(xs), min(xs), min(xs)],
                    [min(ys), min(ys), max(ys), max(ys), min(ys)], linewidth=0.5, color='lightblue')

    cmap = plt.get_cmap('tab20', len(set(cell_to_region.values()))) if cell_to_region else None

    for c in cells.values():
        color = cmap(cell_to_region[c.name]) if cmap and cell_to_region and c.name in cell_to_region else 'none'
        ax.add_patch(patches.Rectangle((c.x, c.y), c.w, c.h,
                                       fill=color != 'none', facecolor=color,
                                       edgecolor='black', linewidth=0.8))
        ax.text(c.x + c.w / 2, c.y + c.h / 2, c.name,
                ha='center', va='center', fontsize=7, color='blue', alpha=0.9)

    ax.set(title=title, xlabel="X", ylabel="Y")
    ax.axis('equal')
    ax.set(xlim=(-10, params.chip_w + 10), ylim=(-10, params.chip_h + 10))
    plt.tight_layout()
    plt.show()


def plot_congestion(
    heat: np.ndarray, params: PlacementParams, title: str = "VLSI Congestion Heatmap"
) -> None:
    plt.figure(figsize=(7, 5))
    plt.imshow(heat, origin='lower', extent=(0, params.chip_w, 0, params.chip_h), aspect='auto')
    plt.colorbar(label='Routing demand (arb)')
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    plt.show()
