import random
from typing import Dict, List, Tuple
from data import Cell

CHIP_W, CHIP_H = 2000, 2000

example_netlist_dict = {
    "cells": {
        "core0": {"w": 60, "h": 60}, "core1": {"w": 60, "h": 60},
        "core2": {"w": 60, "h": 60}, "core3": {"w": 60, "h": 60},
        "icache0": {"w": 100, "h": 80}, "dcache0": {"w": 100, "h": 80},
        "l2cache": {"w": 250, "h": 200}, "l3cache": {"w": 350, "h": 280},
        "noc_router": {"w": 150, "h": 150}, "bus_matrix": {"w": 120, "h": 120},
        "dma": {"w": 80, "h": 80}, "pcie": {"w": 140, "h": 100},
        "ethernet": {"w": 130, "h": 100}, "usb": {"w": 100, "h": 90},
        "pll": {"w": 120, "h": 120, "fixed": True, "x": 50, "y": 50},
        "phy0": {"w": 140, "h": 140, "fixed": True, "x": CHIP_W-200, "y": 50},
        "phy1": {"w": 140, "h": 140, "fixed": True, "x": 50, "y": CHIP_H-200},
        "phy2": {"w": 140, "h": 140, "fixed": True, "x": CHIP_W-200, "y": CHIP_H-200},
    },
    "nets": [
        {"pins": ["core0", "icache0", "dcache0", "noc_router"]},
        {"pins": ["core1", "icache0", "dcache0", "noc_router"]},
        {"pins": ["core2", "icache0", "dcache0", "noc_router"]},
        {"pins": ["core3", "icache0", "dcache0", "noc_router"]},
        {"pins": ["noc_router", "l2cache", "l3cache", "bus_matrix"]},
        {"pins": ["l2cache", "l3cache"]},
        {"pins": ["bus_matrix", "dma", "pcie", "ethernet", "usb"]},
        {"pins": ["pll", "noc_router", "bus_matrix"]},
        {"pins": ["phy0", "pcie", "ethernet"]},
        {"pins": ["phy1", "usb", "dma"]},
        {"pins": ["phy2", "l3cache", "noc_router"]},
    ]
}

def create_cpu_like_blockages(cells: Dict[str, Cell], rng_seed: int = 42) -> List[Tuple[float, float, float, float]]:
    pad_thick = 60
    cross_thick = 100
    cache_w, cache_h = CHIP_W * 0.25, CHIP_H * 0.15

    blockages = [
        # IO pad ring edges
        (0, 0, CHIP_W, pad_thick), (0, CHIP_H - pad_thick, CHIP_W, CHIP_H),
        (0, 0, pad_thick, CHIP_H), (CHIP_W - pad_thick, 0, CHIP_W, CHIP_H),
        # Cache macros reserved area
        (CHIP_W * 0.35, CHIP_H * 0.05, CHIP_W * 0.35 + cache_w, CHIP_H * 0.05 + cache_h),
        (CHIP_W * 0.35, CHIP_H * 0.75, CHIP_W * 0.35 + cache_w, CHIP_H * 0.75 + cache_h),
        # Center cross vertical and horizontal
        (CHIP_W / 2 - cross_thick / 2, 0, CHIP_W / 2 + cross_thick / 2, CHIP_H),
        (0, CHIP_H / 2 - cross_thick / 2, CHIP_W, CHIP_H / 2 + cross_thick / 2)
    ]

    def overlaps_fixed(x0, y0, x1, y1):
        return any(
            c.fixed and x1 > c.x and x0 < c.x + c.w and y1 > c.y and y0 < c.y + c.h
            for c in cells.values()
        )

    return [b for b in blockages if not overlaps_fixed(*b)]
