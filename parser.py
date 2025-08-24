from typing import Dict, List, Tuple, Optional
import random
from data import Cell, Net


def parse_netlist(netlist: Dict) -> Tuple[Dict[str, Cell], List[Net]]:
    cells = {
        name: Cell(
            name=name,
            w=float(info.get("w", 10.0)),
            h=float(info.get("h", 10.0)),
            fixed=bool(info.get("fixed", False)),
            x=float(info.get("x", 0.0)),
            y=float(info.get("y", 0.0)),
            region=tuple(info["region"]) if "region" in info else None,
        )
        for name, info in netlist.get("cells", {}).items()
    }
    nets = [Net(name=n.get("name", f"n{i}"), pins=list(n.get("pins", []))) for i, n in enumerate(netlist.get("nets", []))]
    return cells, nets


def init_placement(
    cells: Dict[str, Cell],
    chip_w: float,
    chip_h: float,
    init_coords: Optional[Dict[str, Tuple[float, float]]],
    rng: random.Random,
) -> None:
    for c in cells.values():
        if c.fixed:
            continue
        if init_coords and c.name in init_coords:
            c.x, c.y = init_coords[c.name]
        else:
            c.x = rng.uniform(0, chip_w - c.w)
            c.y = rng.uniform(0, chip_h - c.h)
