
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def _stable_bucket(value: str, n_buckets: int, salt: str) -> int:
    h = hashlib.md5((salt + "␟" + str(value)).encode("utf-8")).hexdigest()
    return int(h[:8], 16) % max(1, n_buckets)


@dataclass
class _ColumnConfig:
    name: str
    is_high_card: bool
    top_k: int
    n_buckets: int
    salt: str
    top_values: List[str]
    value_to_bucket: Dict[str, int]
    bucket_to_value_counts: Dict[int, Dict[str, int]]
    had_nulls: bool


class HighCardinalityAdapter:
    def __init__(
        self,
        base_synthesizer: Any,
        cardinality_threshold: int = 500,
        top_k: int = 100,
        n_buckets: int = 512,
        alpha: float = 1.0,
        temperature: float = 1.0,
        seed: int = 42,
        columns: Optional[List[str]] = None,
    ) -> None:
        self.base = base_synthesizer
        self.threshold = int(cardinality_threshold)
        self.top_k = int(top_k)
        self.n_buckets = int(n_buckets)
        self.alpha = float(alpha)
        self.temperature = float(temperature)
        self.seed = int(seed)
        self.columns = set(columns) if columns else None

        self._configs: Dict[str, _ColumnConfig] = {}
        self._fitted = False
        random.seed(self.seed)
        np.random.seed(self.seed)

    def _transform_column(self, series: pd.Series, cfg: _ColumnConfig) -> pd.Series:
        s = series.fillna("__NA__").astype(str)

        def enc(v: str) -> str:
            if v == "__NA__":
                return "__NA__"
            if v in cfg.top_values:
                return v
            b = cfg.value_to_bucket.get(v)
            if b is None:
                b = _stable_bucket(v, cfg.n_buckets, cfg.salt)
            return f"__BUCKET_{b}__"

        # PRIMA: return s.map(enc).astype("category")
        return s.map(enc).astype(object)  # <- importante: object, non category

    def fit(self, df: pd.DataFrame, **kwargs) -> "HighCardinalityAdapter":
        df2 = df.copy()
        cat_cols = [c for c in df2.columns if str(df2[c].dtype) in ("object", "category")]
        if self.columns:
            cat_cols = [c for c in cat_cols if c in self.columns]

        for col in cat_cols:
            cfg = self._fit_column(df2, col)
            if cfg.is_high_card:
                df2[col] = self._transform_column(df2[col], cfg)

        self.base.fit(df2, **kwargs)
        self._fitted = True
        return self

    def sample(self, num_rows: int, **kwargs) -> pd.DataFrame:
        self._ensure_fitted()
        sampled = self.base.sample(num_rows=num_rows, **kwargs)
        return self._decode(sampled)

    def sample_from_conditions(self, conditions, **kwargs) -> pd.DataFrame:
        self._ensure_fitted()
        if not hasattr(self.base, "sample_from_conditions"):
            raise AttributeError("Underlying synthesizer does not support conditional sampling.")
        transformed = []
        for cond in conditions:
            cond2 = dict(cond)
            for col, val in list(cond2.items()):
                if col in self._configs and self._configs[col].is_high_card:
                    cond2[col] = self._encode_condition(val, self._configs[col])
            transformed.append(cond2)
        sampled = self.base.sample_from_conditions(conditions=transformed, **kwargs)
        return self._decode(sampled)

    # internals

    def _ensure_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("HighCardinalityAdapter is not fitted. Call .fit(df) first.")

    def _fit_column(self, df: pd.DataFrame, col: str) -> _ColumnConfig:
        s = df[col]
        had_nulls = s.isna().any()
        s_filled = s.fillna("__NA__").astype(str)

        vc = s_filled.value_counts(dropna=False)
        unique = int(vc.shape[0])

        is_high = unique > self.threshold
        salt = f"{col}__{self.seed}"

        top_values: List[str] = []
        value_to_bucket: Dict[str, int] = {}
        bucket_to_value_counts: Dict[int, Dict[str, int]] = {}

        if is_high:
            top_values = vc.head(self.top_k).index.tolist()
            tail_values = [v for v in vc.index if v not in top_values and v != "__NA__"]
            for v in tail_values:
                b = _stable_bucket(v, self.n_buckets, salt)
                value_to_bucket[v] = b
                bucket_to_value_counts.setdefault(b, {})
                bucket_to_value_counts[b][v] = int(vc.get(v, 0))

        cfg = _ColumnConfig(
            name=col,
            is_high_card=is_high,
            top_k=self.top_k,
            n_buckets=self.n_buckets,
            salt=salt,
            top_values=top_values,
            value_to_bucket=value_to_bucket,
            bucket_to_value_counts=bucket_to_value_counts,
            had_nulls=had_nulls,
        )
        self._configs[col] = cfg
        return cfg

    def _transform_column(self, series: pd.Series, cfg: _ColumnConfig) -> pd.Series:
        s = series.fillna("__NA__").astype(str)

        def enc(v: str) -> str:
            if v == "__NA__":
                return "__NA__"
            if v in cfg.top_values:
                return v
            b = cfg.value_to_bucket.get(v)
            if b is None:
                b = _stable_bucket(v, cfg.n_buckets, cfg.salt)
            return f"__BUCKET_{b}__"

        return s.map(enc).astype("category")

    def _encode_condition(self, val, cfg: _ColumnConfig):
        if not cfg.is_high_card:
            return val
        if pd.isna(val):
            return "__NA__"
        val = str(val)
        if val in cfg.top_values or val == "__NA__":
            return val
        b = cfg.value_to_bucket.get(val, _stable_bucket(val, cfg.n_buckets, cfg.salt))
        return f"__BUCKET_{b}__"

    def _decode(self, sampled_df: pd.DataFrame) -> pd.DataFrame:
        df = sampled_df.copy()
        rng = random.Random(self.seed)

        for col, cfg in self._configs.items():
            if not cfg.is_high_card or col not in df.columns:
                continue
            s = df[col].astype(str)

            def decode_one(v: str):
                if v == "__NA__":
                    return np.nan if cfg.had_nulls else None
                if v in cfg.top_values:
                    return v
                if v.startswith("__BUCKET_") and v.endswith("__"):
                    try:
                        b = int(v.replace("__BUCKET_", "").replace("__", ""))
                    except ValueError:
                        return v
                    return self._sample_from_bucket(cfg, b, rng)
                return v

            df[col] = s.map(decode_one)
            if cfg.had_nulls:
                df[col] = df[col].where(pd.notnull(df[col]), np.nan)

        return df

    def _sample_from_bucket(self, cfg: _ColumnConfig, bucket: int, rng: random.Random) -> str:
        counts = cfg.bucket_to_value_counts.get(bucket)
        if not counts:
            all_tail: List[str] = []
            for _b, d in cfg.bucket_to_value_counts.items():
                for v, c in d.items():
                    all_tail.extend([v] * max(1, int(c)))
            return rng.choice(all_tail) if all_tail else f"__BUCKET_{bucket}__"

        items = list(counts.items())
        vals, cts = zip(*items)
        cts = np.asarray(cts, dtype=float) + 1.0  # Laplace smoothing
        # simple softmax with temperature
        logits = np.log(cts) / 1.0
        probs = np.exp(logits - logits.max())
        probs = probs / probs.sum()
        return rng.choices(list(vals), weights=list(probs), k=1)[0]
