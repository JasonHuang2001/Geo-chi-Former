import os
import hashlib
from typing import Any, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler

try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError:
    torch = None

    class Dataset:
        pass

from data.eop_core import (
    DEFAULT_FUTURE_EAM_NOISE_GROWTH,
    DEFAULT_FUTURE_EAM_NOISE_RATIO,
    get_chi_wilson_numpy,
    resolve_future_eam_parameter,
)


AS_TO_MAS = 1000.0
RAD_TO_MAS = (180.0 / np.pi) * 3600.0 * AS_TO_MAS


EAM_REQUIRED_COLS = [
    "aam_x", "aam_y", "aam_vx", "aam_vy",
    "oam_x", "oam_y", "oam_vx", "oam_vy",
    "ham_x", "ham_y", "ham_vx", "ham_vy",
    "slam_x", "slam_y",
]


EAM_AGG_COLS = [
    "aam_x", "aam_y",
    "oam_x", "oam_y",
    "ham_x", "ham_y",
    "slam_x", "slam_y",
    "eam_x", "eam_y",
]


def _cfg(config: Any, key: str, default: Any = None) -> Any:
    return getattr(config, key, default)


def _as_list(value: Any) -> List[float]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _to_float_frame(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    df[list(cols)] = df[list(cols)].interpolate(limit_direction="both").ffill().bfill()
    if df[list(cols)].isna().any().any():
        missing = df[list(cols)].columns[df[list(cols)].isna().any()].tolist()
        raise ValueError(f"EOP loader still contains missing columns after cleaning: {missing}")
    return df


class Dataset_EOP_ULGap(Dataset):
    """
    EOP loader for chi/EAM ultra-low gap experiments.

    Returned tuple:
      0 hist_chi_no_long [seq_len, 2]
      1 hist_eam         [seq_len, len(EAM_AGG_COLS)]
      2 fut_chi_no_long  [pred_len, 2]
      3 fut_eam          [pred_len, len(EAM_AGG_COLS)]
      4 hist_ul_gap      [seq_len, 2] raw physical scale
      5 fut_ul_gap       [pred_len, 2] raw physical scale
      6 hist_time_mark   [seq_len, C_time]
      7 fut_time_mark    [pred_len, C_time]
      8 p0_raw           [2]
      9 chi0_raw         [2]
     10 hist_pm_raw      [seq_len, 2]
     11 fut_pm_raw       [pred_len, 2]
    """

    chi_cols = ["Chi_x_no_long", "Chi_y_no_long"]
    eam_cols_agg = EAM_AGG_COLS.copy()

    def __init__(self, config: Any, flag: str = "train"):
        if flag not in {"train", "val", "test", "all"}:
            raise ValueError(f"Unsupported dataset flag: {flag}")

        self.config = config
        self.flag = flag
        self.seq_len = int(_cfg(config, "seq_len", 96))
        self.pred_len = int(_cfg(config, "pred_len", 30))
        self.root_path = str(_cfg(config, "root_path", "./data/"))
        self.data_path = str(_cfg(config, "data_path", "eop_data_xy_EAM.csv"))
        self.eam_cols = list(_cfg(config, "eam_cols", EAM_AGG_COLS))
        self.dt = float(_cfg(config, "dt", 1.0))
        self.tc = float(_cfg(config, "tc", 433.0))
        self.q = float(_cfg(config, "q", 179.0))
        self.scale_chi = bool(_cfg(config, "scale_chi", True))
        self.scale_eam = bool(_cfg(config, "scale_eam", True))
        self.scale_eam_rad_to_mas = bool(_cfg(config, "scale_eam_rad_to_mas", True))
        self.use_future_observed_eam = bool(_cfg(config, "use_future_observed_eam", False))
        self.future_eam_cols = list(_cfg(config, "future_eam_cols", self.eam_cols))
        self.future_eam_forecast_path = str(_cfg(config, "future_eam_forecast_path", "eam14forecast_daily.csv"))
        self.future_eam_forecast_start_date = pd.Timestamp(
            _cfg(config, "future_eam_forecast_start_date", "2021-05-20")
        ).normalize()
        self.future_eam_available_len = int(_cfg(config, "future_eam_available_len", 14))
        if self.use_future_observed_eam:
            self.future_eam_noise_ratio = resolve_future_eam_parameter(
                _cfg(config, "future_eam_noise_ratio", DEFAULT_FUTURE_EAM_NOISE_RATIO),
                self.future_eam_cols,
                "future_eam_noise_ratio",
            )
            self.future_eam_noise_growth = resolve_future_eam_parameter(
                _cfg(config, "future_eam_noise_growth", DEFAULT_FUTURE_EAM_NOISE_GROWTH),
                self.future_eam_cols,
                "future_eam_noise_growth",
            )
        else:
            self.future_eam_noise_ratio = np.zeros(len(self.future_eam_cols), dtype=np.float32)
            self.future_eam_noise_growth = np.zeros(len(self.future_eam_cols), dtype=np.float32)
        self.future_eam_noise_seed = int(_cfg(config, "future_eam_noise_seed", _cfg(config, "seed", 0)))
        self.future_eam_abs_threshold = float(_cfg(config, "future_eam_abs_threshold", 1e-3))
        self.use_fixed_test_forecast_dates = bool(_cfg(config, "use_fixed_test_forecast_dates", True))
        self.test_forecast_stride = int(_cfg(config, "test_forecast_stride", 7))
        self.ul_gap_window = int(_cfg(config, "ul_gap_window", 1095))
        self.ul_gap_periods = [float(p) for p in _as_list(_cfg(config, "ul_gap_periods", [365.25, 365.25 / 2.0]))]
        self.ul_gap_huber_epsilon = float(_cfg(config, "ul_gap_huber_epsilon", 1.35))
        self.ul_gap_huber_max_iter = int(_cfg(config, "ul_gap_huber_max_iter", 1000))
        self.ul_gap_scaler_max_samples = int(_cfg(config, "ul_gap_scaler_max_samples", 512))
        self.precompute_ul_gap = bool(_cfg(config, "precompute_ul_gap", True))
        self.refresh_ul_gap_cache = bool(_cfg(config, "refresh_ul_gap_cache", False))
        self.ul_gap_cache_dir = str(_cfg(
            config,
            "ul_gap_cache_dir",
            os.path.join(self.root_path, ".cache", "eop_ul_gap"),
        ))
        explicit_cache = _cfg(config, "ul_gap_cache_path", None)
        self.ul_gap_cache_path = str(explicit_cache) if explicit_cache else None
        self.verbose_ul_gap_cache = bool(_cfg(config, "verbose_ul_gap_cache", True))
        self.use_doy_time_mark_features = bool(_cfg(
            config,
            "use_doy_time_mark_features",
            _cfg(config, "use_doy_time_features", False),
        ))
        self.doy_time_periods = [float(p) for p in _as_list(_cfg(
            config,
            "doy_time_periods",
            [365.25, 365.25 / 2.0, 365.25 / 3.0, 365.25 / 4.0],
        ))]

        if self.seq_len <= 0 or self.pred_len <= 0:
            raise ValueError("seq_len and pred_len must be positive")
        if self.ul_gap_window <= 5:
            raise ValueError("ul_gap_window must exceed 5")
        if self.ul_gap_huber_epsilon < 1.0:
            raise ValueError("ul_gap_huber_epsilon must be at least 1.0")
        if self.test_forecast_stride <= 0:
            raise ValueError(f"test_forecast_stride must be positive; got {self.test_forecast_stride}")
        if self.future_eam_available_len < 0:
            raise ValueError(f"future_eam_available_len cannot be negative; got {self.future_eam_available_len}")
        if np.any(self.future_eam_noise_ratio < 0):
            raise ValueError(f"future_eam_noise_ratio cannot be negative: {self.future_eam_noise_ratio}")
        if np.any(self.future_eam_noise_growth < 0):
            raise ValueError(f"future_eam_noise_growth cannot be negative: {self.future_eam_noise_growth}")
        if self.use_future_observed_eam and self.future_eam_cols != self.eam_cols:
            raise ValueError(
                "future_eam_cols must match eam_cols; "
                f"future_eam_cols={self.future_eam_cols}, eam_cols={self.eam_cols}"
            )

        self._read_data()
        self._prepare_future_eam_inputs()
        self._resolve_split_dates()
        self.samples = self._build_samples_for_flag(flag)
        self._prepare_ul_gap_cache()
        self._fit_or_load_scalers()

    def _read_data(self) -> None:
        path = os.path.join(self.root_path, self.data_path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"EOP/EAM data file not found: {path}")

        df_raw = pd.read_csv(path)
        required = ["date", "xpole", "ypole"] + EAM_REQUIRED_COLS
        missing = [col for col in required if col not in df_raw.columns]
        if missing:
            raise ValueError(f"EOP/EAM input is missing columns: {missing}")

        df_raw = df_raw[required].copy()
        df_raw["date"] = pd.to_datetime(df_raw["date"]).dt.normalize()
        df_raw = df_raw.sort_values("date").drop_duplicates("date").reset_index(drop=True)
        df_raw = _to_float_frame(df_raw, ["xpole", "ypole"] + EAM_REQUIRED_COLS)
        if len(df_raw) < self.ul_gap_window + self.pred_len + 3:
            raise ValueError("Insufficient data for UL-gap history and forecast windows")

        pm_x = df_raw["xpole"].to_numpy(dtype=np.float64) * AS_TO_MAS
        pm_y = df_raw["ypole"].to_numpy(dtype=np.float64) * AS_TO_MAS
        chi_x, chi_y, pm_x_aligned, pm_y_aligned = get_chi_wilson_numpy(
            pm_x,
            pm_y,
            dt=self.dt,
            tc=self.tc,
            q=self.q,
        )

        df = df_raw.iloc[1:-1].copy().reset_index(drop=True)
        df["xpole"] = pm_x_aligned
        df["ypole"] = pm_y_aligned
        df["Chi_x"] = chi_x
        df["Chi_y"] = chi_y
        if self.scale_eam_rad_to_mas:
            for col in EAM_REQUIRED_COLS:
                df[col] = df[col].astype(float) * RAD_TO_MAS

        self.df = df.reset_index(drop=True)
        self.dates = self.df["date"].to_numpy()
        self.date_index = pd.to_datetime(self.dates)
        self.t_days = (self.date_index - self.date_index[0]).days.to_numpy(dtype=np.float64)
        self.pm_raw = self.df[["xpole", "ypole"]].to_numpy(dtype=np.float32)
        self.chi_raw = self.df[["Chi_x", "Chi_y"]].to_numpy(dtype=np.float32)
        self.eam_raw = self._build_eam_agg(self.df, self.eam_cols).astype(np.float32)
        self.eam_total = self._build_eam_agg(self.df, ["eam_x", "eam_y"]).astype(np.float32)
        self.gap_raw = (self.chi_raw - self.eam_total).astype(np.float32)
        self.time_marks = self._build_time_marks(self.date_index)

    @staticmethod
    def _build_eam_agg_frame(df: pd.DataFrame) -> pd.DataFrame:
        values = {
            "aam_x": df["aam_x"].to_numpy(dtype=np.float64) + df["aam_vx"].to_numpy(dtype=np.float64),
            "aam_y": df["aam_y"].to_numpy(dtype=np.float64) + df["aam_vy"].to_numpy(dtype=np.float64),
            "oam_x": df["oam_x"].to_numpy(dtype=np.float64) + df["oam_vx"].to_numpy(dtype=np.float64),
            "oam_y": df["oam_y"].to_numpy(dtype=np.float64) + df["oam_vy"].to_numpy(dtype=np.float64),
            "ham_x": df["ham_x"].to_numpy(dtype=np.float64) + df["ham_vx"].to_numpy(dtype=np.float64),
            "ham_y": df["ham_y"].to_numpy(dtype=np.float64) + df["ham_vy"].to_numpy(dtype=np.float64),
            "slam_x": df["slam_x"].to_numpy(dtype=np.float64),
            "slam_y": df["slam_y"].to_numpy(dtype=np.float64),
        }
        values["eam_x"] = values["aam_x"] + values["oam_x"] + values["ham_x"] + values["slam_x"]
        values["eam_y"] = values["aam_y"] + values["oam_y"] + values["ham_y"] + values["slam_y"]
        return pd.DataFrame(values, index=df.index)

    @staticmethod
    def _build_eam_agg(df: pd.DataFrame, cols: Iterable[str] = None) -> np.ndarray:
        agg = Dataset_EOP_ULGap._build_eam_agg_frame(df)
        cols = EAM_AGG_COLS if cols is None else list(cols)
        missing = [col for col in cols if col not in agg.columns]
        if missing:
            raise ValueError(f"Unsupported aggregate EAM columns: {missing}")
        return agg[cols].to_numpy(dtype=np.float64)

    @staticmethod
    def _clean_forecast_eam_frame(
        df: pd.DataFrame,
        cols: Iterable[str],
        group_col: str,
        abs_threshold: float,
    ) -> pd.DataFrame:
        frame = df.copy()
        cols = list(cols)
        for col in cols:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        if abs_threshold is not None and abs_threshold > 0:
            frame[cols] = frame[cols].mask(frame[cols].abs() > abs_threshold, np.nan)

        groups = []
        for _, group in frame.groupby(group_col, sort=False):
            group = group.copy()
            group[cols] = group[cols].interpolate(limit_direction="both").ffill().bfill()
            groups.append(group)
        frame = pd.concat(groups, axis=0).sort_index() if groups else frame
        if frame[cols].isna().any().any():
            missing = frame[cols].columns[frame[cols].isna().any()].tolist()
            raise ValueError(f"EAM forecast still contains missing columns after cleaning: {missing}")
        return frame

    def _prepare_future_eam_inputs(self) -> None:
        self.future_eam_forecast_lookup = {}
        self.future_eam_col_indices = []
        self.future_eam_noise_std = None
        if not self.use_future_observed_eam:
            return

        missing = [col for col in self.future_eam_cols if col not in self.eam_cols]
        if missing:
            raise ValueError(f"future_eam_cols are absent from eam_cols: {missing}")
        self.future_eam_col_indices = [self.eam_cols.index(col) for col in self.future_eam_cols]

        future_eam = self.eam_raw[:, self.future_eam_col_indices]
        finite_values = future_eam[np.isfinite(future_eam).all(axis=1)]
        if finite_values.size == 0:
            raise ValueError("Observed EAM contains no finite rows")
        self.future_eam_noise_std = (
            np.nanstd(finite_values, axis=0).astype(np.float32) * self.future_eam_noise_ratio
        )
        self._load_future_eam_forecast()

    def _load_future_eam_forecast(self) -> None:
        forecast_path = os.path.join(self.root_path, self.future_eam_forecast_path)
        if not os.path.exists(forecast_path):
            raise FileNotFoundError(f"Future EAM forecast not found: {forecast_path}")

        df_raw = pd.read_csv(forecast_path)
        required = ["issue_date", "date", "lead_day"] + EAM_REQUIRED_COLS
        missing = [col for col in required if col not in df_raw.columns]
        if missing:
            raise ValueError(f"Future EAM forecast is missing columns: {missing}")

        df_raw = df_raw[required].copy()
        df_raw["issue_date"] = pd.to_datetime(df_raw["issue_date"]).dt.normalize()
        df_raw["date"] = pd.to_datetime(df_raw["date"]).dt.normalize()
        df_raw["lead_day"] = pd.to_numeric(df_raw["lead_day"], errors="coerce")
        df_raw = df_raw.dropna(subset=["issue_date", "date", "lead_day"]).reset_index(drop=True)
        df_raw["lead_day"] = df_raw["lead_day"].astype(int)

        forecast = df_raw[["issue_date", "date", "lead_day"]].copy()
        agg = self._build_eam_agg_frame(df_raw)
        for col in self.future_eam_cols:
            forecast[col] = agg[col].to_numpy(dtype=np.float64)
        forecast = self._clean_forecast_eam_frame(
            forecast,
            self.future_eam_cols,
            group_col="issue_date",
            abs_threshold=self.future_eam_abs_threshold,
        )
        if self.scale_eam_rad_to_mas:
            forecast[self.future_eam_cols] = forecast[self.future_eam_cols].astype(float) * RAD_TO_MAS

        if not forecast.empty:
            first_issue = pd.Timestamp(forecast["issue_date"].min()).normalize()
            if self.future_eam_forecast_start_date < first_issue:
                self.future_eam_forecast_start_date = first_issue

        self.future_eam_forecast_lookup = {}
        for _, row in forecast.iterrows():
            key = (pd.Timestamp(row["issue_date"]).normalize(), int(row["lead_day"]))
            self.future_eam_forecast_lookup[key] = row[self.future_eam_cols].to_numpy(dtype=np.float32)

    def _future_eam_noise_scale(self, lead_day: int) -> np.ndarray:
        if self.future_eam_noise_std is None:
            return np.zeros(len(self.future_eam_cols), dtype=np.float32)
        growth = 1.0 + self.future_eam_noise_growth * np.float32(lead_day)
        return self.future_eam_noise_std * np.float32(growth)

    def _future_eam_window_for_sample(self, issue_idx: int, f0: int, f1: int) -> np.ndarray:
        if not self.use_future_observed_eam:
            return self.eam_raw[f0:f1].astype(np.float32)

        issue_date = pd.Timestamp(self.date_index[issue_idx]).normalize()
        rng = np.random.default_rng(self.future_eam_noise_seed + int(issue_date.strftime("%Y%m%d")))
        rows = []
        for lead_day, valid_idx in enumerate(range(f0, f1)):
            if lead_day >= self.future_eam_available_len:
                rows.append(np.zeros(len(self.future_eam_cols), dtype=np.float32))
                continue

            forecast_key = (issue_date, int(lead_day))
            if issue_date >= self.future_eam_forecast_start_date:
                forecast = self.future_eam_forecast_lookup.get(forecast_key)
                if forecast is None:
                    rows.append(np.zeros(len(self.future_eam_cols), dtype=np.float32))
                else:
                    rows.append(forecast.astype(np.float32))
                continue

            future_eam = self.eam_raw[valid_idx, self.future_eam_col_indices].astype(np.float32)
            if np.any(self.future_eam_noise_ratio > 0):
                future_eam = future_eam + rng.normal(
                    loc=0.0,
                    scale=self._future_eam_noise_scale(lead_day),
                    size=future_eam.shape,
                ).astype(np.float32)
            rows.append(future_eam)

        future_eam = np.stack(rows, axis=0).astype(np.float32)
        if not np.isfinite(future_eam).all():
            raise ValueError(f"Future EAM contains non-finite values for {issue_date.date()}")
        return future_eam

    def _build_time_marks(self, dates: pd.DatetimeIndex) -> np.ndarray:
        if not self.use_doy_time_mark_features:
            self.time_mark_names = ["month", "day", "weekday"]
            return np.vstack([
                dates.month.to_numpy(dtype=np.float32),
                dates.day.to_numpy(dtype=np.float32),
                dates.weekday.to_numpy(dtype=np.float32),
            ]).T.astype(np.float32)

        doy = dates.dayofyear.to_numpy(dtype=np.float64) - 1.0
        marks = []
        names = []
        for period in self.doy_time_periods:
            if period <= 0:
                raise ValueError(f"Invalid day-of-year period: {period}")
            tag = f"{period:.6g}".replace(".", "p")
            angle = 2.0 * np.pi * doy / period
            marks.extend([np.sin(angle), np.cos(angle)])
            names.extend([f"doy_sin_{tag}", f"doy_cos_{tag}"])
        self.time_mark_names = names
        return np.vstack(marks).T.astype(np.float32)

    def _resolve_split_dates(self) -> None:
        self.train_start_date = pd.Timestamp(_cfg(self.config, "train_start_date", "1993-01-01")).normalize()
        val_start = pd.Timestamp(_cfg(self.config, "val_start_date", "2016-01-01")).normalize()
        test_start = pd.Timestamp(_cfg(self.config, "test_forecast_start_date", "2020-01-02")).normalize()

        train_end = _cfg(self.config, "train_end_date", None)
        val_end = _cfg(self.config, "val_end_date", None)
        test_end = _cfg(self.config, "test_forecast_end_date", None)

        self.val_start_date = val_start
        self.test_start_date = test_start
        self.train_end_date = (
            pd.Timestamp(train_end).normalize()
            if train_end is not None
            else self.val_start_date - pd.Timedelta(days=1)
        )
        self.val_end_date = (
            pd.Timestamp(val_end).normalize()
            if val_end is not None
            else self.test_start_date - pd.Timedelta(days=1)
        )
        self.test_end_date = (
            pd.Timestamp(test_end).normalize()
            if test_end is not None
            else pd.Timestamp(self.date_index[-1]).normalize()
        )

    def _build_samples_for_flag(self, flag: str) -> np.ndarray:
        if flag == "train":
            return self._build_samples_for_range(self.train_start_date, self.train_end_date)
        if flag == "val":
            return self._build_samples_for_range(self.val_start_date, self.val_end_date)
        if flag == "test":
            if self.use_fixed_test_forecast_dates:
                return self._build_fixed_test_samples()
            return self._build_samples_for_range(self.test_start_date, self.test_end_date)
        samples = [
            i for i in range(len(self.df))
            if self._sample_is_valid(i)
        ]
        return self._finalize_samples(samples, "all")

    def _build_samples_for_range(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> np.ndarray:
        samples = []
        for i, date in enumerate(self.date_index):
            issue_date = pd.Timestamp(date).normalize()
            if issue_date < start_date or issue_date > end_date:
                continue
            if not self._sample_is_valid(i):
                continue
            future_end = pd.Timestamp(self.date_index[i + self.pred_len - 1]).normalize()
            if future_end <= end_date:
                samples.append(i)
        return self._finalize_samples(samples, f"{start_date.date()}~{end_date.date()}")

    def _build_fixed_test_samples(self) -> np.ndarray:
        date_to_idx = {
            pd.Timestamp(date).normalize(): idx
            for idx, date in enumerate(self.date_index)
        }
        issue_dates = pd.date_range(
            self.test_start_date,
            self.test_end_date,
            freq=f"{self.test_forecast_stride}D",
        )
        samples = []
        forecast_windows = []
        missing = []

        for issue_date in issue_dates:
            issue_date = pd.Timestamp(issue_date).normalize()
            fut_end_date = issue_date + pd.Timedelta(days=self.pred_len - 1)
            issue_idx = date_to_idx.get(issue_date)
            fut_end_idx = date_to_idx.get(fut_end_date)

            if issue_idx is None:
                missing.append(f"{issue_date.date()}: issue date is missing")
                continue
            if fut_end_idx is None:
                missing.append(f"{issue_date.date()}: labels are missing through {fut_end_date.date()}")
                continue
            if not self._sample_is_valid(issue_idx):
                missing.append(f"{issue_date.date()}: insufficient history or UL-gap context")
                continue
            if fut_end_idx - issue_idx + 1 != self.pred_len:
                missing.append(f"{issue_date.date()}: forecast labels are not {self.pred_len} continuous days")
                continue

            samples.append(issue_idx)
            forecast_windows.append(np.asarray(self.date_index[issue_idx:issue_idx + self.pred_len], dtype="datetime64[ns]"))

        if missing:
            preview = "\n".join(missing[:5])
            extra = "" if len(missing) <= 5 else f"\n... {len(missing) - 5} more issue dates failed"
            raise ValueError(f"Fixed test issue dates could not be constructed:\n{preview}{extra}")

        finalized = self._finalize_samples(samples, f"fixed-test {self.test_start_date.date()}~{self.test_end_date.date()}")
        self.issue_dates = np.asarray([self.date_index[idx].to_datetime64() for idx in finalized])
        self.forecast_date_windows = np.asarray(forecast_windows)
        return finalized

    def _sample_is_valid(self, issue_idx: int) -> bool:
        return (
            issue_idx - self.seq_len >= 0
            and issue_idx - self.ul_gap_window >= 0
            and issue_idx + self.pred_len <= len(self.df)
        )

    @staticmethod
    def _finalize_samples(samples: List[int], label: str) -> np.ndarray:
        if len(samples) == 0:
            raise RuntimeError(f"EOP loader produced no samples for {label}")
        return np.asarray(samples, dtype=np.int64)

    def _design_matrix(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=np.float64)
        cols = [t]
        for period in self.ul_gap_periods:
            if period <= 0:
                raise ValueError(f"Invalid UL-gap period: {period}")
            angle = 2.0 * np.pi * t / period
            cols.extend([np.sin(angle), np.cos(angle)])
        return np.vstack(cols).T

    def _fit_ul_linear(self, issue_idx: int, out_start: int, out_end: int) -> np.ndarray:
        fit_start = issue_idx - self.ul_gap_window
        fit_end = issue_idx
        t_fit = self.t_days[fit_start:fit_end]
        x_fit = self._design_matrix(t_fit)
        t_out = self.t_days[out_start:out_end]
        out = np.zeros((out_end - out_start, 2), dtype=np.float32)

        for channel in range(2):
            y_fit = self.gap_raw[fit_start:fit_end, channel].astype(np.float64)
            model = HuberRegressor(
                epsilon=self.ul_gap_huber_epsilon,
                alpha=0.0,
                fit_intercept=True,
                max_iter=self.ul_gap_huber_max_iter,
            )
            model.fit(x_fit, y_fit)
            out[:, channel] = (
                float(model.intercept_) + float(model.coef_[0]) * t_out
            ).astype(np.float32)
        return out

    def _ul_gap_cache_path(self) -> str:
        if self.ul_gap_cache_path:
            return os.path.abspath(self.ul_gap_cache_path)
        data_file = os.path.abspath(os.path.join(self.root_path, self.data_path))
        stat = os.stat(data_file)
        payload = "|".join([
            "v2",
            data_file,
            str(stat.st_size),
            str(int(stat.st_mtime_ns)),
            self.flag,
            str(self.seq_len),
            str(self.pred_len),
            str(self.ul_gap_window),
            ",".join(f"{p:.12g}" for p in self.ul_gap_periods),
            str(self.ul_gap_huber_epsilon),
            str(self.ul_gap_huber_max_iter),
            ",".join(str(int(x)) for x in self.samples.tolist()),
        ])
        digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
        name = (
            f"ul_gap_{self.flag}_L{self.seq_len}_P{self.pred_len}_"
            f"W{self.ul_gap_window}_{digest}.npz"
        )
        return os.path.join(self.ul_gap_cache_dir, name)

    def _build_ul_gap_cache_array(self) -> np.ndarray:
        cache = np.empty((len(self.samples), self.seq_len + self.pred_len, 2), dtype=np.float32)
        for sample_pos, issue_idx in enumerate(self.samples):
            issue_idx = int(issue_idx)
            h0 = issue_idx - self.seq_len
            f1 = issue_idx + self.pred_len
            cache[sample_pos] = self._fit_ul_linear(issue_idx=issue_idx, out_start=h0, out_end=f1)
        return cache

    def _prepare_ul_gap_cache(self) -> None:
        self.ul_gap_cache = None
        if not self.precompute_ul_gap:
            return

        cache_path = self._ul_gap_cache_path()
        if self.ul_gap_cache_path and not os.path.exists(cache_path):
            raise FileNotFoundError(f"Explicit UL-gap cache not found: {cache_path}")
        if not self.ul_gap_cache_path:
            os.makedirs(self.ul_gap_cache_dir, exist_ok=True)
        if os.path.exists(cache_path) and not self.refresh_ul_gap_cache:
            with np.load(cache_path) as data:
                samples = data["samples"].astype(np.int64)
                ul_gap = data["ul_gap"].astype(np.float32)
            expected_shape = (len(self.samples), self.seq_len + self.pred_len, 2)
            if np.array_equal(samples, self.samples) and ul_gap.shape == expected_shape:
                self.ul_gap_cache = ul_gap
                return
            if self.ul_gap_cache_path and ul_gap.shape[1:] == expected_shape[1:]:
                positions = {int(sample): index for index, sample in enumerate(samples)}
                missing = [int(sample) for sample in self.samples if int(sample) not in positions]
                if not missing:
                    self.ul_gap_cache = np.stack(
                        [ul_gap[positions[int(sample)]] for sample in self.samples], axis=0
                    ).astype(np.float32)
                    return
                raise ValueError(f"Explicit UL-gap cache is missing sample indices: {missing[:5]}")

        if self.ul_gap_cache_path:
            raise ValueError(f"Explicit UL-gap cache is incompatible with this dataset: {cache_path}")

        if self.verbose_ul_gap_cache:
            print(
                f"[*] Precomputing UL-gap cache: flag={self.flag}, "
                f"samples={len(self.samples)}, path={cache_path}"
            )
        self.ul_gap_cache = self._build_ul_gap_cache_array()
        np.savez(cache_path, samples=self.samples.astype(np.int64), ul_gap=self.ul_gap_cache)

    def _get_ul_gap_for_sample(self, sample_index: int, issue_idx: int, h0: int, f1: int) -> np.ndarray:
        if self.ul_gap_cache is not None:
            return self.ul_gap_cache[sample_index]
        return self._fit_ul_linear(issue_idx=issue_idx, out_start=h0, out_end=f1)

    def _sample_raw(self, sample_index: int) -> Tuple[np.ndarray, ...]:
        issue_idx = int(self.samples[sample_index])
        h0 = issue_idx - self.seq_len
        h1 = issue_idx
        f0 = issue_idx
        f1 = issue_idx + self.pred_len
        ul = self._get_ul_gap_for_sample(sample_index, issue_idx, h0, f1)
        hist_ul = ul[:self.seq_len].astype(np.float32)
        fut_ul = ul[self.seq_len:].astype(np.float32)
        hist_chi_no_long = (self.chi_raw[h0:h1] - hist_ul).astype(np.float32)
        fut_chi_no_long = (self.chi_raw[f0:f1] - fut_ul).astype(np.float32)
        fut_eam = self._future_eam_window_for_sample(issue_idx, f0, f1)
        return (
            hist_chi_no_long,
            self.eam_raw[h0:h1].astype(np.float32),
            fut_chi_no_long,
            fut_eam,
            hist_ul,
            fut_ul,
            self.time_marks[h0:h1].astype(np.float32),
            self.time_marks[f0:f1].astype(np.float32),
            self.pm_raw[h1 - 1].astype(np.float32),
            self.chi_raw[h1 - 1].astype(np.float32),
            self.pm_raw[h0:h1].astype(np.float32),
            self.pm_raw[f0:f1].astype(np.float32),
        )

    def _selected_scaler_samples(self, samples: np.ndarray) -> np.ndarray:
        if self.ul_gap_scaler_max_samples <= 0 or len(samples) <= self.ul_gap_scaler_max_samples:
            return samples
        positions = np.linspace(0, len(samples) - 1, self.ul_gap_scaler_max_samples).round().astype(int)
        return samples[positions]

    def _fit_or_load_scalers(self) -> None:
        self.chi_scaler = StandardScaler()
        self.eam_scaler = StandardScaler()
        if not self.scale_chi and not self.scale_eam:
            return

        chi_mean = _cfg(self.config, "ul_gap_chi_scaler_mean", None)
        chi_scale = _cfg(self.config, "ul_gap_chi_scaler_scale", None)
        eam_mean = _cfg(self.config, "ul_gap_eam_scaler_mean", None)
        eam_scale = _cfg(self.config, "ul_gap_eam_scaler_scale", None)

        restored_chi = False
        restored_eam = False
        if self.scale_chi and chi_mean is not None and chi_scale is not None:
            self._restore_scaler(self.chi_scaler, chi_mean, chi_scale)
            restored_chi = True
        if self.scale_eam and eam_mean is not None and eam_scale is not None:
            self._restore_scaler(self.eam_scaler, eam_mean, eam_scale)
            restored_eam = True
        if (not self.scale_chi or restored_chi) and (not self.scale_eam or restored_eam):
            return

        train_samples = self._build_samples_for_range(self.train_start_date, self.train_end_date)
        selected_samples = self._selected_scaler_samples(train_samples)

        if self.scale_chi and not restored_chi:
            chi_rows = []
            for issue_idx in selected_samples:
                old_samples = self.samples
                old_cache = self.ul_gap_cache
                self.samples = np.asarray([issue_idx], dtype=np.int64)
                self.ul_gap_cache = None
                hist_chi, _, fut_chi, *_ = self._sample_raw(0)
                self.samples = old_samples
                self.ul_gap_cache = old_cache
                chi_rows.extend([hist_chi, fut_chi])
            chi_train = np.concatenate(chi_rows, axis=0)
            self.chi_scaler.fit(chi_train)
            setattr(self.config, "ul_gap_chi_scaler_mean", self.chi_scaler.mean_.tolist())
            setattr(self.config, "ul_gap_chi_scaler_scale", self.chi_scaler.scale_.tolist())

        if self.scale_eam and not restored_eam:
            train_mask = (
                (self.date_index >= self.train_start_date)
                & (self.date_index <= self.train_end_date)
            )
            eam_train = self.eam_raw[train_mask]
            if len(eam_train) == 0:
                raise RuntimeError("The training-only EAM scaler has no usable data")

            self.eam_scaler.fit(eam_train)
            setattr(self.config, "ul_gap_eam_scaler_mean", self.eam_scaler.mean_.tolist())
            setattr(self.config, "ul_gap_eam_scaler_scale", self.eam_scaler.scale_.tolist())

    @staticmethod
    def _restore_scaler(scaler: StandardScaler, mean: Any, scale: Any) -> None:
        scaler.mean_ = np.asarray(mean, dtype=np.float64)
        scaler.scale_ = np.asarray(scale, dtype=np.float64)
        scaler.var_ = scaler.scale_ ** 2
        scaler.n_features_in_ = int(scaler.mean_.shape[0])

    def _transform_chi(self, values: np.ndarray) -> np.ndarray:
        if not self.scale_chi:
            return values.astype(np.float32)
        return self.chi_scaler.transform(values.reshape(-1, 2)).reshape(values.shape).astype(np.float32)

    def _transform_eam(self, values: np.ndarray) -> np.ndarray:
        if not self.scale_eam:
            return values.astype(np.float32)
        eam_dim = len(self.eam_cols)
        return self.eam_scaler.transform(values.reshape(-1, eam_dim)).reshape(values.shape).astype(np.float32)

    def inverse_transform_chi(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32)
        if not self.scale_chi:
            return arr
        return self.chi_scaler.inverse_transform(arr.reshape(-1, 2)).reshape(arr.shape).astype(np.float32)

    def inverse_transform_eam(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=np.float32)
        if not self.scale_eam:
            return arr
        eam_dim = len(self.eam_cols)
        return self.eam_scaler.inverse_transform(arr.reshape(-1, eam_dim)).reshape(arr.shape).astype(np.float32)

    def sample_hist_dates(self, sample_index: int) -> np.ndarray:
        issue_idx = int(self.samples[sample_index])
        return self.dates[issue_idx - self.seq_len:issue_idx]

    def sample_future_dates(self, sample_index: int) -> np.ndarray:
        issue_idx = int(self.samples[sample_index])
        return self.dates[issue_idx:issue_idx + self.pred_len]

    def __getitem__(self, index: int):
        if torch is None:
            raise ModuleNotFoundError("EOP_loader.__getitem__ requires torch, but torch is not installed.")

        (
            hist_chi_no_long,
            hist_eam,
            fut_chi_no_long,
            fut_eam,
            hist_ul_gap,
            fut_ul_gap,
            hist_time_mark,
            fut_time_mark,
            p0_raw,
            chi0_raw,
            hist_pm_raw,
            fut_pm_raw,
        ) = self._sample_raw(index)

        item = (
            self._transform_chi(hist_chi_no_long),
            self._transform_eam(hist_eam),
            self._transform_chi(fut_chi_no_long),
            self._transform_eam(fut_eam),
            hist_ul_gap,
            fut_ul_gap,
            hist_time_mark,
            fut_time_mark,
            p0_raw,
            chi0_raw,
            hist_pm_raw,
            fut_pm_raw,
        )
        return tuple(torch.tensor(value, dtype=torch.float32) for value in item)

    def __len__(self) -> int:
        return int(len(self.samples))


EOP_loader = Dataset_EOP_ULGap
