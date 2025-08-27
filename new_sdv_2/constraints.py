from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import pandas as pd

@dataclass
class RangeConstraint:
    column: str
    min_val: Optional[float] = None
    max_val: Optional[float] = None

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if self.column in out.columns:
            if self.min_val is not None:
                out[self.column] = np.where(out[self.column] < self.min_val, self.min_val, out[self.column])
            if self.max_val is not None:
                out[self.column] = np.where(out[self.column] > self.max_val, self.max_val, out[self.column])
        return out

@dataclass
class UniqueConstraint:
    column: str

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        return (
            df.drop_duplicates(subset=[self.column], keep="first").reset_index(drop=True)
            if self.column in df.columns else df
        )

def apply_constraints(df: pd.DataFrame, constraints: List[object]) -> pd.DataFrame:
    out = df.copy()
    for c in constraints or []:
        if hasattr(c, "apply"):
            out = c.apply(out)
    return out
