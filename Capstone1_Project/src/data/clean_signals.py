# src/data/clean_signals.py
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

from .empatica_loader import load_empatica

TARGET_FS = 4.0  # Hz
ROLL_MED_SEC = 3
ROLL_MEAN_SEC = 3

def read_cfg():
    with open("configs/base.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _smooth_series(s: pd.Series, fs: float) -> pd.Series:
    """Median then mean smoothing in native sampling rate."""
    med = int(max(1, ROLL_MED_SEC * fs))
    mean = int(max(1, ROLL_MEAN_SEC * fs))
    s1 = s.rolling(window=med, center=True, min_periods=1).median()
    s2 = s1.rolling(window=mean, center=True, min_periods=1).mean()
    return s2

def _resample_to_grid(df: pd.DataFrame, col: str, grid_ts: np.ndarray) -> pd.Series:
    """
    df: has 'timestamp' (float seconds) + value column 'col'
    grid_ts: numpy array of target timestamps (seconds)
    returns: Series aligned to DatetimeIndex(grid_ts)
    """
    target_idx = pd.to_datetime(grid_ts, unit="s")
    small = df[["timestamp", col]].dropna().sort_values("timestamp")
    if small.empty:
        return pd.Series(index=target_idx, dtype=float)

    s = pd.Series(
        small[col].to_numpy(),
        index=pd.to_datetime(small["timestamp"].to_numpy(), unit="s"),
    )
    s = s.sort_index().reindex(s.index.union(target_idx)).interpolate(method="time")
    s = s.reindex(target_idx)
    return s

def process_session(session_dir: Path) -> pd.DataFrame:
    """
    session_dir e.g. .../Wearable_Dataset/STRESS/S01
    Returns aligned 4 Hz dataframe with columns:
      timestamp, EDA, TEMP, HR, BVP, ACC_mag
    """
    # Load available signals
    eda  = load_empatica(session_dir / "EDA.csv")  if (session_dir / "EDA.csv").exists()  else None
    temp = load_empatica(session_dir / "TEMP.csv") if (session_dir / "TEMP.csv").exists() else None
    hr   = load_empatica(session_dir / "HR.csv")   if (session_dir / "HR.csv").exists()   else None
    bvp  = load_empatica(session_dir / "BVP.csv")  if (session_dir / "BVP.csv").exists()  else None
    acc  = load_empatica(session_dir / "ACC.csv")  if (session_dir / "ACC.csv").exists()  else None

    # Smooth in native rate
    if eda is not None and "value" in eda:
        fs = float(eda["sample_rate_hz"].iloc[0]) if "sample_rate_hz" in eda else TARGET_FS
        eda["value"] = _smooth_series(eda["value"], fs).clip(lower=0, upper=60)

    if temp is not None and "value" in temp:
        fs = float(temp["sample_rate_hz"].iloc[0]) if "sample_rate_hz" in temp else TARGET_FS
        temp["value"] = _smooth_series(temp["value"], fs)

    if hr is not None and "value" in hr:
        fs = float(hr["sample_rate_hz"].iloc[0]) if "sample_rate_hz" in hr else TARGET_FS
        hr["value"] = _smooth_series(hr["value"], fs)

    if bvp is not None and "value" in bvp:
        fs = float(bvp["sample_rate_hz"].iloc[0]) if "sample_rate_hz" in bvp else TARGET_FS
        bvp["value"] = _smooth_series(bvp["value"], fs)

    if acc is not None and {"x","y","z"}.issubset(acc.columns):
        fs = float(acc["sample_rate_hz"].iloc[0]) if "sample_rate_hz" in acc else TARGET_FS
        acc["mag"] = np.sqrt(acc["x"]**2 + acc["y"]**2 + acc["z"]**2)
        acc["mag"] = _smooth_series(acc["mag"], fs)

    # Build common time window
    ts_min, ts_max = None, None
    for d in (eda, temp, hr, bvp, acc):
        if d is None or "timestamp" not in d: 
            continue
        mn, mx = d["timestamp"].min(), d["timestamp"].max()
        ts_min = mn if ts_min is None else min(ts_min, mn)
        ts_max = mx if ts_max is None else max(ts_max, mx)

    if ts_min is None or ts_max is None or not np.isfinite([ts_min, ts_max]).all():
        return pd.DataFrame()  # nothing to align

    # Common 4 Hz grid
    grid = np.arange(ts_min, ts_max, 1.0 / TARGET_FS)
    target_idx = pd.to_datetime(grid, unit="s")
    res = pd.DataFrame({"timestamp": grid}, index=target_idx)

    # Resample each signal to the grid
    if eda is not None and "value" in eda:
        res["EDA"] = _resample_to_grid(eda.rename(columns={"value": "EDA"}), "EDA", grid)
    if temp is not None and "value" in temp:
        res["TEMP"] = _resample_to_grid(temp.rename(columns={"value": "TEMP"}), "TEMP", grid)
    if hr is not None and "value" in hr:
        res["HR"] = _resample_to_grid(hr.rename(columns={"value": "HR"}), "HR", grid)
    if bvp is not None and "value" in bvp:
        res["BVP"] = _resample_to_grid(bvp.rename(columns={"value": "BVP"}), "BVP", grid)
    if acc is not None and "mag" in acc:
        res["ACC_mag"] = _resample_to_grid(acc.rename(columns={"mag": "ACC_mag"}), "ACC_mag", grid)

    # Return with timestamp column (index is only for alignment)
    return res.reset_index(drop=True)

def main():
    cfg = read_cfg()
    raw_root = Path(cfg["data"]["raw_dir"])
    out_root = Path(cfg["data"]["processed_dir"]) / "clean"
    out_root.mkdir(parents=True, exist_ok=True)

    records = []
    for condition in ["STRESS", "AEROBIC", "ANAEROBIC"]:
        cdir = raw_root / condition
        if not cdir.exists():
            continue
        for subj in sorted([d for d in cdir.iterdir() if d.is_dir()]):
            try:
                df = process_session(subj)
            except Exception as e:
                print(f"[WARN] Skipping {condition}/{subj.name}: {e}")
                continue

            if df.empty:
                print(f"[INFO] No usable data in {condition}/{subj.name}, skipping.")
                continue

            subj_out_dir = out_root / condition
            subj_out_dir.mkdir(parents=True, exist_ok=True)
            outp = subj_out_dir / f"{subj.name}.csv"
            df.to_csv(outp, index=False)

            # quick features
            feats = {"condition": condition, "subject": subj.name}
            for col in ["EDA", "TEMP", "HR", "BVP", "ACC_mag"]:
                if col in df.columns:
                    vals = df[col].to_numpy(dtype=float)
                    feats[f"{col}_mean"] = float(np.nanmean(vals))
                    feats[f"{col}_std"]  = float(np.nanstd(vals))
            records.append(feats)
            print("Saved:", outp)

    if records:
        feat_df = pd.DataFrame(records)
        feat_df.to_csv(Path(cfg["data"]["processed_dir"]) / "features_per_session.csv", index=False)
        print("Saved features:", Path(cfg["data"]["processed_dir"]) / "features_per_session.csv")

if __name__ == "__main__":
    main()
