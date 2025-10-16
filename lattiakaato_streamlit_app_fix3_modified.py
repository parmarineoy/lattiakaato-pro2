import io, math
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


# ---------------- Korkoeron käsittely ----------------

def adjust_other_area_heights(shower_area: Shower, other_area_points: List[Tuple[float, float, float]], height_diff_mm: float) -> List[Tuple[float, float, float]]:
    """
    Korjaa muun alueen korkopisteet suihkualueeseen nähden annetun korkoeron perusteella.
    Jos korkoero > 0, lisätään se muun alueen pisteisiin (kynnys).
    Jos korkoero = 0, ei muuteta pisteitä (vesi saa virrata suihkualueelle).
    """
    if height_diff_mm > 0:
        return [(x, y, z + height_diff_mm / 1000.0) for x, y, z in other_area_points]
    else:
        return other_area_points

# ---------------- Geometria ----------------

def distance_point_to_rect(px: float, py: float, rect: tuple[float, float, float, float]) -> float:
    x0, y0, w, h = rect
    x1, y1 = x0 + w, y0 + h
    cx = min(max(px, x0), x1)
    cy = min(max(py, y0), y1)
    return math.hypot(px - cx, py - cy)

def closest_point_on_rect(px: float, py: float, rect: tuple[float, float, float, float]) -> tuple[float, float]:
    x0, y0, w, h = rect
    x1, y1 = x0 + w, y0 + h
    cx = min(max(px, x0), x1)
    cy = min(max(py, y0), y1)
    return cx, cy

def inside_rect(px: float, py: float, rect: tuple[float, float, float, float]) -> bool:
    x0, y0, w, h = rect
    return (x0 <= px <= x0 + w) and (y0 <= py <= y0 + h)

def signed_side_of_line(px: float, py: float, line: tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = line
    return (x1 - x0) * (py - y0) - (y1 - y0) * (px - x0)

def distance_point_to_segment(px: float, py: float, x0: float, y0: float, x1: float, y1: float) -> float:
    vx, vy = x1 - x0, y1 - y0
    wx, wy = px - x0, py - y0
    seg_len2 = vx * vx + vy * vy
    if seg_len2 == 0:
        return math.hypot(px - x0, py - y0)
    t = (wx * vx + wy * vy) / seg_len2
    t = max(0.0, min(1.0, t))
    projx = x0 + t * vx
    projy = y0 + t * vy
    return math.hypot(px - projx, py - projy)

# ---------------- Data ----------------

@dataclass
class Shower:
    rect: Tuple[float, float, float, float]
    drain_x: float
    drain_y: float

@dataclass
class SplitLine:
    line: Tuple[float, float, float, float]
    left_target_type: Optional[str]
    left_target_index: Optional[int]
    right_target_type: Optional[str]
    right_target_index: Optional[int]

@dataclass
class Config:
    L: float; W: float; grid: float
    showers: List[Shower]
    drains: List[Tuple[float, float]]
    step_mm: float
    k_s_min: float; k_o_min: float
    threshold: Optional[Tuple[float, float, float, float]]
    thr_H_mm: float
    thr_k_min: float
    split_lines: List[SplitLine]
    layers_m: float
