"""Core data structures shared by every stage of the pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np


@dataclass
class Study:
    """One imaging study in canonical RAS+ orientation.

    Volumes are indexed [x, y, z] with +x = patient right, +y = anterior, +z = superior. 2D images are indexed [row, col] in display orientation.
    """

    kind: str  # 'ct' | 'mr' | 'xray' | 'image2d' | 'other'
    pixels: np.ndarray  # HU for CT
    spacing: tuple[float, ...]  # mm, per array axis
    affine: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)  # ingestion-level QC codes

    @property
    def voxel_volume_mm3(self) -> float:
        return float(np.prod(self.spacing))


class Segmentation:
    """Integer label map plus a name→label table (TotalSegmentator naming)."""

    def __init__(self, labels: np.ndarray, names: dict[str, int], source: str = "unknown"):
        self.labels = labels
        self.names = dict(names)
        self.source = source
        present = np.unique(labels)
        self._present = set(int(v) for v in present if v != 0)

    def has(self, name: str) -> bool:
        return self.names.get(name, -1) in self._present

    def mask(self, name: str) -> np.ndarray:
        idx = self.names.get(name)
        if idx is None or idx not in self._present:
            return np.zeros(self.labels.shape, dtype=bool)
        return self.labels == idx

    def union(self, names) -> np.ndarray:
        ids = [self.names[n] for n in names if self.names.get(n) in self._present]
        if not ids:
            return np.zeros(self.labels.shape, dtype=bool)
        return np.isin(self.labels, ids)

    def names_matching(self, prefix: str) -> list[str]:
        return [n for n in self.names if n.startswith(prefix) and self.has(n)]


@dataclass
class Measurement:
    """A single quantitative result with its own reliability statement."""

    key: str
    value: float | None
    unit: str
    reliability: str = "high"  # 'high' | 'moderate' | 'low' | 'invalid'
    notes: list[str] = field(default_factory=list)  # machine-readable caveat codes
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        if d["value"] is not None:
            d["value"] = round(float(d["value"]), 2)
        return d


@dataclass
class Finding:
    """A clinically phrased, evidence-linked statement derived from measurements."""

    id: str
    domain: str
    severity: int  # 0 normal/info, 1 attention, 2 significant, 3 urgent review
    title_ar: str
    title_en: str
    detail_ar: str
    detail_en: str
    recommendation_ar: str = ""
    recommendation_en: str = ""
    evidence: list[str] = field(default_factory=list)  # keys into rules.EVIDENCE
    reliability: str = "high"
    caveats_ar: list[str] = field(default_factory=list)
    measurement_keys: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)
