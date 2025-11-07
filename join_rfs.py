import pandas as pd, hashlib, io, os

# ===== 1) PASTE THE SAME SALT YOU USE IN THE APP =====
SALT = "maryhadalittlelambandtonyhadabigelephant"

# ===== 2) Filenames (kept in the same folder as this script) =====
APPS_PATH   = "applications_sample.csv"   # or .xlsx is fine
SCORED_PATH = "rfs_scored.csv"

# ===== 3) Column names in your applications file =====
EMAIL_COL = "Email"             # change if your header is different
DATE_COL  = "ApplicationDate"   # optional; set to None if you don't have it

def norm_email(x):
    if pd.isna(x): return ""
    return str(x).strip().lower()

def pid_from_email(email):
    return hashlib.sha256((SALT + norm_email(email)).encode("utf-8")).hexdigest()[:16]

def read_table(path: str) -> pd.DataFrame:
    """Robust reader for CSV/XLSX."""
    lower = path.lower()
    if lower.endswith((".xlsx", ".xls")):
        import openpyxl  # ensure dependency exists
        return pd.read_excel(path, engine="openpyxl")
    # CSV: try encodings and sniff delimiter
    encs = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    for enc in encs:
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=enc)
        except Exception:
            pass
    # last resort
    return pd.read_csv(path, encoding="latin1")

# ---- 4) Load files ----
apps   = read_table(APPS_PATH)
scored = read_table(SCORED_PATH)

if EMAIL_COL not in apps.columns:
    raise SystemExit(f"Couldn't find '{EMAIL_COL}' in {APPS_PATH}. Edit EMAIL_COL at top of script.")

# ---- 5) Compute PID on the applications file (same as the app) ----
apps["PID"] = apps[EMAIL_COL].map(pid_from_email)

# Optional: collapse duplicates by latest application date
if DATE_COL and DATE_COL in apps.columns:
    apps["_date"] = pd.to_datetime(apps[DATE_COL], errors="coerce")
    apps = apps.sort_values("_date").drop_duplicates("PID", keep="last")

# ---- 6) Join to the scored table ----
keep_cols = ["PID", "RFS", "Decision"]
missing = [c for c in keep_cols if c not in scored.columns]
if missing:
    raise SystemExit(f"{SCORED_PATH} is missing columns: {missing}")

out = apps.merge(scored[keep_cols], on="PID", how="left")

# Tidy & sort
front = [EMAIL_COL, "PID", "Decision", "RFS"]
rest  = [c for c in out.columns if c not in front]
out   = out[front + rest]
out   = out.sort_values(["Decision", "RFS"], ascending=[True, False])

# ---- 7) Save contact list ----
out.to_csv("contact_list.csv", index=False, encoding="utf-8")
matched = out["RFS"].notna().sum()
unmatched = out.shape[0] - matched
print(f"Saved contact_list.csv  |  matched: {matched}  unmatched: {unmatched}")
