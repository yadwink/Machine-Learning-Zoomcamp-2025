from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

def _first_token(line: str) -> str:
    # Handle lines like "2013-06-12 16:18:58,2013-06-12 16:18:58,2013-06-12 16:18:58"
    return line.split(",")[0].strip()

def _parse_start_time(token: str) -> float:
    # Try epoch float first
    try:
        return float(token)
    except ValueError:
        pass
    # Try "YYYY-MM-DD HH:MM:SS"
    try:
        dt = datetime.strptime(token, "%Y-%m-%d %H:%M:%S")
        return dt.timestamp()
    except ValueError:
        # Fallback to pandas parser
        dt = pd.to_datetime(token, errors="raise", utc=False)
        if getattr(dt, "tzinfo", None) is not None:
            dt = dt.tz_localize(None)
        return dt.timestamp()

def load_empatica(csv_path: str | Path) -> pd.DataFrame:
    """
    Robust loader for Empatica E4 CSVs in this dataset.

    ACC -> columns: x,y,z,timestamp,sample_rate_hz,start_time_utc
    IBI -> columns: t,ibi,timestamp,start_time_utc   (no sample_rate_hz)
    TAGS-> columns: utc,timestamp
    Others (EDA/HR/TEMP/BVP) -> value,timestamp,sample_rate_hz,start_time_utc
    """
    p = Path(csv_path)
    stem = p.stem.upper()

    with open(p, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
        second_line = f.readline().strip()

    # IBI: first line = start time; no sample rate on line 2
    if stem == "IBI":
        start = _parse_start_time(_first_token(first_line))
        df = pd.read_csv(p, skiprows=1, header=None)
        if df.shape[1] < 2:
            raise ValueError(f"Unexpected IBI format in {p}")
        df = df.iloc[:, :2]
        df.columns = ["t", "ibi"]
        df["t"] = pd.to_numeric(df["t"], errors="coerce")
        df["ibi"] = pd.to_numeric(df["ibi"], errors="coerce")
        df["timestamp"] = start + df["t"].astype(float)
        df["start_time_utc"] = start
        return df

    # TAGS: one timestamp per line
    if stem == "TAGS":
        raw = pd.read_csv(p, header=None, names=["utc"], skip_blank_lines=True)
        def _to_epoch(x):
            s = str(x).strip()
            try:
                return float(s)
            except ValueError:
                dt = pd.to_datetime(s, errors="coerce", utc=False)
                if pd.isna(dt): 
                    return np.nan
                if getattr(dt, "tzinfo", None) is not None:
                    dt = dt.tz_localize(None)
                return dt.timestamp()
        raw["timestamp"] = raw["utc"].map(_to_epoch)
        return raw

    # General case (EDA/HR/TEMP/BVP/ACC)
    start = _parse_start_time(_first_token(first_line))
    fs_token = _first_token(second_line)
    try:
        fs = float(fs_token)
    except ValueError:
        fs = None

    df = pd.read_csv(p, skiprows=2, header=None)
    if stem == "ACC":
        if df.shape[1] < 3:
            raise ValueError(f"ACC expected 3 columns in {p}, got {df.shape[1]}")
        df = df.iloc[:, :3]
        df.columns = ["x", "y", "z"]
    else:
        df = df.iloc[:, :1]
        df.columns = ["value"]

    if fs and fs > 0:
        n = len(df)
        df["timestamp"] = start + (pd.RangeIndex(n) / fs)
        df["sample_rate_hz"] = fs
    else:
        df["timestamp"] = np.nan
        df["sample_rate_hz"] = np.nan

    df["start_time_utc"] = start
    return df
