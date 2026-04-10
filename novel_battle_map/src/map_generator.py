from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle, Ellipse, FancyArrowPatch, Rectangle
from scipy.interpolate import make_interp_spline

from src.models import BattleExtractionResult, BattleEvent, MilitaryEntity


@dataclass
class MapContext:
    river_path: Optional[np.ndarray]
    road_path: Optional[np.ndarray]
    bridge_pos: Optional[Tuple[float, float]]
    hill_points: List[Tuple[float, float]]
    village_pos: Optional[Tuple[float, float]]
    forest_centers: List[Tuple[float, float]]
    unit_positions: Dict[str, Tuple[float, float]]


class ProceduralBattleMapGenerator:
    def __init__(self, seed: int = 42, width: int = 100, height: int = 75) -> None:
        self.seed = seed
        self.width = width
        self.height = height
        self.rng = random.Random(seed)

    def generate(self, extraction: BattleExtractionResult, output_path: Path) -> None:
        fig, ax = plt.subplots(figsize=(16, 12), dpi=100)
        fig.patch.set_facecolor("#efe8d2")
        ax.set_facecolor("#efe8d2")

        X, Y, Z, hills = self._build_terrain(extraction)
        self._draw_terrain(ax, X, Y, Z)

        ctx = MapContext(
            river_path=None,
            road_path=None,
            bridge_pos=None,
            hill_points=hills,
            village_pos=None,
            forest_centers=[],
            unit_positions={},
        )

        if extraction.terrain_elements.has_river:
            ctx.river_path = self._draw_river(ax)
        if extraction.terrain_elements.has_forest:
            ctx.forest_centers = self._draw_forests(ax)
        if extraction.terrain_elements.has_road:
            ctx.road_path = self._draw_road(ax, preferred=ctx.river_path)
        if extraction.terrain_elements.has_bridge and ctx.river_path is not None:
            ctx.bridge_pos = self._draw_bridge(ax, ctx.river_path)
        if extraction.terrain_elements.has_village:
            ctx.village_pos = self._draw_village(ax, near=ctx.road_path)

        ctx.unit_positions = self._place_units(ax, extraction.military_entities, ctx)
        self._draw_tactical_overlays(ax, extraction.events, ctx)

        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(extraction.chapter_title, fontsize=18, fontweight="bold", color="#2f2a20")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        fig.savefig(output_path, dpi=100)
        plt.close(fig)

    def _build_terrain(self, extraction: BattleExtractionResult) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[float, float]]]:
        x = np.linspace(0, self.width, 220)
        y = np.linspace(0, self.height, 180)
        X, Y = np.meshgrid(x, y)

        Z = np.zeros_like(X)
        hill_points: List[Tuple[float, float]] = []
        bumps = 5 if extraction.terrain_elements.has_hill else 3
        for _ in range(bumps):
            cx = self.rng.uniform(10, self.width - 10)
            cy = self.rng.uniform(10, self.height - 10)
            hill_points.append((cx, cy))
            amp = self.rng.uniform(0.6, 1.8) * (1.3 if extraction.terrain_elements.has_hill else 0.9)
            sx = self.rng.uniform(8, 20)
            sy = self.rng.uniform(8, 20)
            Z += amp * np.exp(-(((X - cx) ** 2) / (2 * sx**2) + ((Y - cy) ** 2) / (2 * sy**2)))

        Z += 0.1 * np.sin(X / 10.0) + 0.07 * np.cos(Y / 8.0)
        return X, Y, Z, hill_points

    def _draw_terrain(self, ax: plt.Axes, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> None:
        ax.imshow(Z, origin="lower", extent=[0, self.width, 0, self.height], cmap="YlOrBr", alpha=0.20)
        ax.contour(X, Y, Z, levels=14, colors="#8c6a43", linewidths=0.8, alpha=0.6)

    def _smooth_path(self, points: np.ndarray, samples: int = 300) -> np.ndarray:
        t = np.arange(points.shape[0])
        k = min(3, points.shape[0] - 1)
        spline_x = make_interp_spline(t, points[:, 0], k=k)
        spline_y = make_interp_spline(t, points[:, 1], k=k)
        tt = np.linspace(0, points.shape[0] - 1, samples)
        return np.column_stack([spline_x(tt), spline_y(tt)])

    def _draw_river(self, ax: plt.Axes) -> np.ndarray:
        pts = np.array(
            [
                [self.rng.uniform(6, 16), self.height - 2],
                [self.width * 0.33, self.height * 0.65 + self.rng.uniform(-8, 8)],
                [self.width * 0.58, self.height * 0.4 + self.rng.uniform(-7, 7)],
                [self.width - self.rng.uniform(8, 16), 2],
            ]
        )
        curve = self._smooth_path(pts)
        ax.plot(curve[:, 0], curve[:, 1], color="#3f7fbf", linewidth=6, alpha=0.75, solid_capstyle="round")
        ax.plot(curve[:, 0], curve[:, 1], color="#80add8", linewidth=3, alpha=0.85)
        return curve

    def _draw_forests(self, ax: plt.Axes) -> List[Tuple[float, float]]:
        centers: List[Tuple[float, float]] = []
        for _ in range(self.rng.randint(2, 4)):
            cx = self.rng.uniform(15, self.width - 15)
            cy = self.rng.uniform(12, self.height - 12)
            centers.append((cx, cy))
            for _ in range(18):
                ex = cx + self.rng.uniform(-8, 8)
                ey = cy + self.rng.uniform(-5, 5)
                e = Ellipse((ex, ey), width=self.rng.uniform(1.5, 3.5), height=self.rng.uniform(1.5, 3.2), angle=self.rng.uniform(0, 180), color="#4e7f45", alpha=0.45)
                ax.add_patch(e)
        return centers

    def _draw_road(self, ax: plt.Axes, preferred: Optional[np.ndarray] = None) -> np.ndarray:
        if preferred is not None and len(preferred) > 20:
            pivot = preferred[len(preferred) // 2]
        else:
            pivot = np.array([self.width * 0.5, self.height * 0.5])
        pts = np.array(
            [
                [5, self.rng.uniform(10, self.height - 10)],
                [pivot[0] - self.rng.uniform(8, 15), pivot[1] + self.rng.uniform(-12, 12)],
                [pivot[0] + self.rng.uniform(8, 15), pivot[1] + self.rng.uniform(-12, 12)],
                [self.width - 5, self.rng.uniform(8, self.height - 8)],
            ]
        )
        curve = self._smooth_path(pts)
        ax.plot(curve[:, 0], curve[:, 1], color="#6e6e6e", linewidth=3.0, alpha=0.8)
        ax.plot(curve[:, 0], curve[:, 1], color="#b5b5b5", linewidth=1.2, alpha=0.9)
        return curve

    def _draw_bridge(self, ax: plt.Axes, river: np.ndarray) -> Tuple[float, float]:
        idx = len(river) // 2
        bx, by = river[idx]
        angle = math.degrees(math.atan2(river[min(idx + 2, len(river) - 1), 1] - river[max(idx - 2, 0), 1], river[min(idx + 2, len(river) - 1), 0] - river[max(idx - 2, 0), 0]))
        bridge = Rectangle((bx - 2.8, by - 0.7), 5.6, 1.4, angle=angle + 90, color="#6d4c2f", alpha=0.9)
        ax.add_patch(bridge)
        ax.text(bx + 1.5, by + 1.2, "Bridge", fontsize=9, color="#4b3220")
        return float(bx), float(by)

    def _draw_village(self, ax: plt.Axes, near: Optional[np.ndarray] = None) -> Tuple[float, float]:
        if near is not None and len(near) > 30:
            anchor = near[self.rng.randint(20, len(near) - 20)]
            vx = float(np.clip(anchor[0] + self.rng.uniform(-8, 8), 10, self.width - 10))
            vy = float(np.clip(anchor[1] + self.rng.uniform(-6, 6), 8, self.height - 8))
        else:
            vx = self.rng.uniform(10, self.width - 10)
            vy = self.rng.uniform(8, self.height - 8)

        for i in range(5):
            house = Rectangle((vx + i * 0.9, vy + (i % 2) * 0.8), 1.2, 0.9, color="#b58a62", alpha=0.9)
            ax.add_patch(house)
        ax.text(vx + 1.5, vy + 2.2, "Village", fontsize=9, color="#5f4630")
        return vx, vy

    def _place_units(self, ax: plt.Axes, entities: List[MilitaryEntity], ctx: MapContext) -> Dict[str, Tuple[float, float]]:
        positions: Dict[str, Tuple[float, float]] = {}
        if not entities:
            return positions

        red_slots = [(12, 14), (16, 22), (20, 28), (24, 34)]
        blue_slots = [(self.width - 12, self.height - 14), (self.width - 18, self.height - 22), (self.width - 24, self.height - 30), (self.width - 28, self.height - 36)]
        neutral_slots = [(self.width * 0.5, 15), (self.width * 0.55, self.height - 15), (self.width * 0.4, self.height * 0.5)]

        red_i = blue_i = neutral_i = 0
        for ent in entities[:12]:
            if ent.side == "red":
                x, y = red_slots[red_i % len(red_slots)]
                red_i += 1
                color = "#bf3b3b"
            elif ent.side == "blue":
                x, y = blue_slots[blue_i % len(blue_slots)]
                blue_i += 1
                color = "#2c5aa8"
            else:
                x, y = neutral_slots[neutral_i % len(neutral_slots)]
                neutral_i += 1
                color = "#313131"

            if ctx.bridge_pos and "bridge" in ent.name.lower():
                x, y = ctx.bridge_pos[0] + self.rng.uniform(-4, 4), ctx.bridge_pos[1] + self.rng.uniform(-4, 4)

            ax.add_patch(Circle((x, y), radius=1.3, color=color, alpha=0.95, zorder=6))
            ax.text(x + 1.5, y + 1.2, ent.name[:22], fontsize=8, color="#151515", zorder=7)
            positions[ent.name] = (float(x), float(y))
        return positions

    def _draw_tactical_overlays(self, ax: plt.Axes, events: List[BattleEvent], ctx: MapContext) -> None:
        for event in events[:18]:
            start = ctx.unit_positions.get(event.actor)
            if start is None:
                start = (self.rng.uniform(12, self.width - 12), self.rng.uniform(10, self.height - 10))

            target = self._infer_target_point(event, start, ctx)
            color = "#912c2c" if event.event_type in {"advance", "attack", "engage", "cross", "occupy", "surround"} else "#4b4b4b"

            if event.event_type in {"retreat", "withdraw"}:
                ax.plot([start[0], target[0]], [start[1], target[1]], linestyle="--", linewidth=2.2, color="#5a5a5a", alpha=0.9, zorder=5)
            else:
                arr = FancyArrowPatch(start, target, arrowstyle="-|>", mutation_scale=14, linewidth=2.2, color=color, alpha=0.9, zorder=5)
                ax.add_patch(arr)

            if event.event_type in {"engage", "attack", "defend"}:
                mx, my = (start[0] + target[0]) / 2, (start[1] + target[1]) / 2
                ax.add_patch(Circle((mx, my), radius=2.8, color="#9f3028", alpha=0.18, zorder=4))
                ax.plot([mx - 1.4, mx + 1.4], [my - 1.4, my + 1.4], color="#7f1f1f", linewidth=1.3)
                ax.plot([mx - 1.4, mx + 1.4], [my + 1.4, my - 1.4], color="#7f1f1f", linewidth=1.3)

            if event.event_type == "surround":
                ax.add_patch(Arc(target, width=11, height=8, theta1=20, theta2=340, color="#8a2626", linewidth=1.8, alpha=0.8))

    def _infer_target_point(self, event: BattleEvent, start: Tuple[float, float], ctx: MapContext) -> Tuple[float, float]:
        hint = (event.location_hint or "") + " " + (event.terrain_hint or "")
        hint_lower = hint.lower()

        if ("bridge" in hint_lower or "桥" in hint) and ctx.bridge_pos:
            return ctx.bridge_pos
        if ("river" in hint_lower or "河" in hint) and ctx.river_path is not None:
            pt = ctx.river_path[len(ctx.river_path) // 2]
            return float(pt[0]), float(pt[1])
        if ("forest" in hint_lower or "森林" in hint) and ctx.forest_centers:
            return ctx.forest_centers[0]
        if ("ridge" in hint_lower or "hill" in hint_lower or "高地" in hint or "山脊" in hint) and ctx.hill_points:
            return ctx.hill_points[0]
        if ("village" in hint_lower or "村" in hint) and ctx.village_pos:
            return ctx.village_pos

        if event.event_type in {"retreat", "withdraw"} and ctx.hill_points:
            return ctx.hill_points[-1]
        if event.event_type in {"advance", "attack", "engage", "cross"} and ctx.river_path is not None:
            pt = ctx.river_path[min(len(ctx.river_path) - 1, len(ctx.river_path) // 2 + 25)]
            return float(pt[0]), float(pt[1])

        tx = float(np.clip(start[0] + self.rng.uniform(-18, 18), 5, self.width - 5))
        ty = float(np.clip(start[1] + self.rng.uniform(-14, 14), 5, self.height - 5))
        return tx, ty
