"""
bmespecimen_parser.py
---------------------
Parse Bosch .bmespecimen files (JSON-like) into structured pandas DataFrames
for machine learning and analysis.

Usage:
    python bmespecimen_parser.py your_file.bmespecimen
"""

import json
import re
import pandas as pd
from pathlib import Path


# ------------------------------------------------------------
# JSON LOADING & REPAIR
# ------------------------------------------------------------
def try_load_json(path):
    """Try to load the .bmespecimen file as JSON with minor repairs."""
    text = Path(path).read_text(errors="replace")
    try:
        return json.loads(text)
    except Exception as e:
        # Try to repair trailing commas and similar issues
        repaired = re.sub(r",\s*([\]}])", r"\1", text)
        try:
            return json.loads(repaired)
        except Exception:
            # Try extracting main JSON object
            first, last = text.find("{"), text.rfind("}")
            if first != -1 and last != -1:
                candidate = text[first:last+1]
                return json.loads(candidate)
            raise RuntimeError(f"Failed to parse JSON from {path}")


# ------------------------------------------------------------
# HELPERS TO FIND MEASUREMENT SERIES
# ------------------------------------------------------------
def is_numeric(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def dict_has_numeric_values(d):
    if not isinstance(d, dict):
        return False
    for val in d.values():
        if is_numeric(val):
            return True
        if isinstance(val, str):
            try:
                float(val)
                return True
            except Exception:
                pass
    return False


def find_measurement_series(obj, path=""):
    """
    Recursively search for lists that look like measurement series:
    - list of dicts where each dict has numeric values
    - list of basic numeric types
    Returns list of tuples: (path, list_object)
    """
    series = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_path = f"{path}.{k}" if path else k
            series.extend(find_measurement_series(v, new_path))
    elif isinstance(obj, list):
        if len(obj) == 0:
            return series

        # List of dicts with numeric fields
        if all(isinstance(x, dict) for x in obj[:min(100, len(obj))]):
            if any(dict_has_numeric_values(x) for x in obj[:min(100, len(obj))]):
                series.append((path, obj))
            else:
                for i, x in enumerate(obj[:20]):
                    series.extend(find_measurement_series(x, f"{path}[{i}]"))

        # List of pure numbers
        elif all(isinstance(x, (int, float)) for x in obj[:min(200, len(obj))]):
            series.append((path, obj))

        # Mixed types, search deeper
        else:
            for i, x in enumerate(obj[:50]):
                series.extend(find_measurement_series(x, f"{path}[{i}]"))

    return series


# ------------------------------------------------------------
# METADATA EXTRACTION
# ------------------------------------------------------------
def extract_metadata(root):
    """Extract useful specimen and session metadata."""
    md = {}
    if not isinstance(root, dict):
        return md

    # File-level metadata
    if "meta" in root and isinstance(root["meta"], dict):
        md.update({f"meta_{k}": v for k, v in root["meta"].items() if not isinstance(v, (dict, list))})

    specimen = None
    if "data" in root and isinstance(root["data"], dict):
        if "specimenData" in root["data"]:
            specimen = root["data"]["specimenData"]
        elif "specimen" in root["data"]:
            specimen = root["data"]["specimen"]

    # fallback
    if specimen is None:
        for k in ["specimenData", "specimen", "data"]:
            if k in root and isinstance(root[k], dict):
                specimen = root[k]
                break

    # Extract simple fields
    if specimen and isinstance(specimen, dict):
        for k, v in specimen.items():
            if isinstance(v, (str, int, float, bool)):
                md[f"specimen_{k}"] = v

    # Session info
    if "data" in root and isinstance(root["data"], dict):
        session = root["data"].get("measurementSession")
        if isinstance(session, dict):
            for k, v in session.items():
                if isinstance(v, (str, int, float, bool)):
                    md[f"session_{k}"] = v

        # Board config
        board = root["data"].get("boardConfig")
        if isinstance(board, dict):
            for k, v in board.items():
                if isinstance(v, (str, int, float, bool)):
                    md[f"board_{k}"] = v

    return md


# ------------------------------------------------------------
# MAIN PARSER FUNCTION
# ------------------------------------------------------------
def parse_bmespecimen(path):
    """Return combined pandas DataFrame with measurement data and metadata."""
    data = try_load_json(path)
    series = find_measurement_series(data)
    metadata = extract_metadata(data)

    dfs = []
    for s_path, s in series:
        if not isinstance(s, list) or len(s) == 0:
            continue

        # List of dicts with numeric values
        if isinstance(s[0], dict):
            df = pd.json_normalize(s)
            drop_cols = [c for c in df.columns if df[c].apply(lambda x: isinstance(x, (dict, list))).any()]
            if drop_cols:
                df = df.drop(columns=drop_cols)
            for c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="ignore")
            df["_source_path"] = s_path

        # List of pure numbers
        elif all(isinstance(x, (int, float)) for x in s):
            df = pd.DataFrame({"value": s, "_source_path": s_path})

        else:
            continue

        # Attach metadata
        for mk, mv in metadata.items():
            df[mk] = mv

        dfs.append(df)

    if not dfs:
        print("⚠️ No measurement series found.")
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True, sort=False)
    return combined


# ------------------------------------------------------------
# CLI USAGE
# ------------------------------------------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python bmespecimen_parser.py your_file.bmespecimen")
        sys.exit(1)

    in_path = Path(sys.argv[1])
    df = parse_bmespecimen(in_path)
    out_path = in_path.with_suffix(".csv")
    df.to_csv(out_path, index=False)
    print(f"\n✅ Parsed {in_path.name}")
    print(f"   Found {len(df)} rows and {len(df.columns)} columns.")
    print(f"   Saved CSV to: {out_path}")
