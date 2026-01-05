# src/data/index_dataset.py
from pathlib import Path
import pandas as pd, yaml

def read_cfg():
    with open("configs/base.yaml") as f: return yaml.safe_load(f)

def build_manifest(root: Path) -> pd.DataFrame:
    rows=[]
    for condition in ["STRESS","AEROBIC","ANAEROBIC"]:
        cdir = root/condition
        if not cdir.exists(): continue
        for subj in cdir.iterdir():
            if not subj.is_dir(): continue
            for f in subj.glob("*.csv"):
                rows.append({"condition":condition,"subject":subj.name,"file":f.name,"path":str(f.resolve())})
    return pd.DataFrame(rows)

if __name__=="__main__":
    cfg = read_cfg()
    root = Path(cfg["data"]["raw_dir"])
    out = Path(cfg["data"]["processed_dir"]); out.mkdir(parents=True, exist_ok=True)
    mf = build_manifest(root); mf.to_csv(out/"manifest.csv", index=False)
    print("Saved:", out/"manifest.csv")
