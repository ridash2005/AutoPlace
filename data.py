from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

BBox = Tuple[float, float, float, float]

@dataclass
class Cell:
    name: str
    w: float
    h: float
    fixed: bool = False
    x: float = 0.0
    y: float = 0.0
    region: Optional[BBox] = None

@dataclass
class Net:
    name: str
    pins: List[str]

@dataclass
class PlacementParams:
    chip_w: float
    chip_h: float
    num_partitions: int = 4
    balance_tolerance: float = 0.1
    blockages: List[BBox] = field(default_factory=list)
    keepout: float = 0.0
    anneal_iters: int = 10_000
    rng_seed: int = 42
    visualize: bool = True
