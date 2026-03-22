import os
import ast
from typing import Tuple, Dict, List, Optional

import numpy as np
import pandas as pd
import wfdb
from sklearn.preprocessing import LabelEncoder


# --------- CONFIG --------- #

BASE_PATH = os.path.join("data", "ptbxl")

DATABASE_CSV = os.path.join(BASE_PATH, "ptbxl_database.csv")
SCP_CSV = os.path.join(BASE_PATH, "scp_statements.csv")

TARGET_SAMPLING_RATE = 500          # Hz
TARGET_LENGTH = 5000                # 10 seconds * 500 Hz
OUT_PATH = os.path.join("processed", "ptbxl_500hz_10s.npz")

# If True:
#   keep only records whose mapped diagnostic SCP codes all belong to ONE superclass
# If False:
#   choose the highest-likelihood mapped superclass (your original behaviour)
STRICT_SINGLE_SUPERCLASS = True


# --------- HELPER FUNCTIONS --------- #

def load_metadata() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load PTB-XL metadata and SCP statement information."""
    if not os.path.exists(DATABASE_CSV):
        raise FileNotFoundError(f"Could not find {DATABASE_CSV}")
    if not os.path.exists(SCP_CSV):
        raise FileNotFoundError(f"Could not find {SCP_CSV}")

    df = pd.read_csv(DATABASE_CSV)
    scp_df = pd.read_csv(SCP_CSV, index_col=0)
    return df, scp_df


def build_diagnostic_mapping(scp_df: pd.DataFrame) -> Dict[str, str]:
    """
    Build mapping from SCP code -> diagnostic superclass.

    Keeps only diagnostic SCP statements that have a diagnostic_class.
    """
    diag_df = scp_df[scp_df["diagnostic"] == 1].copy()
    diag_df = diag_df.dropna(subset=["diagnostic_class"])

    mapping = diag_df["diagnostic_class"].to_dict()
    if not mapping:
        raise RuntimeError("Diagnostic mapping is empty – check scp_statements.csv format.")
    return mapping


def parse_scp_codes(scp_str: str) -> Dict[str, float]:
    """Parse a PTB-XL scp_codes string safely into a dict."""
    parsed = ast.literal_eval(scp_str)
    if not isinstance(parsed, dict):
        return {}
    return parsed


def choose_superclass_label(
    scp_str: str,
    scp_mapping: Dict[str, str],
    strict_single_superclass: bool = True,
) -> Optional[str]:
    """
    Convert one row's scp_codes dict into a single diagnostic superclass label.

    Modes:
    - strict_single_superclass=True:
        Keep only if all mapped diagnostic codes belong to the same superclass.
        Example:
          {"AMI": 80, "IMI": 50} -> both may map to MI -> keep MI
          {"NORM": 90, "STTC": 80} -> conflicting superclasses -> drop
    - strict_single_superclass=False:
        Pick mapped superclass of highest-likelihood diagnostic SCP code.
    """
    scp_dict = parse_scp_codes(scp_str)
    diag_items = [(code, likelihood) for code, likelihood in scp_dict.items() if code in scp_mapping]

    if not diag_items:
        return None

    if not strict_single_superclass:
        best_code, _ = max(diag_items, key=lambda t: t[1])
        return scp_mapping[best_code]

    mapped_superclasses = {scp_mapping[code] for code, _ in diag_items}
    if len(mapped_superclasses) == 1:
        return next(iter(mapped_superclasses))
    return None


def extract_superclass_labels(
    df: pd.DataFrame,
    scp_mapping: Dict[str, str],
    strict_single_superclass: bool = True,
) -> pd.Series:
    """
    Convert each row's scp_codes dict into a single diagnostic superclass label.
    """
    labels: List[Optional[str]] = []
    for scp_str in df["scp_codes"]:
        label = choose_superclass_label(
            scp_str,
            scp_mapping,
            strict_single_superclass=strict_single_superclass,
        )
        labels.append(label)

    return pd.Series(labels, name="diagnostic_superclass")


def load_and_process_signal(record_path: str) -> np.ndarray:
    """
    Load a single 12-lead ECG using wfdb and:
    - transpose to shape (leads, time)
    - crop or pad to TARGET_LENGTH
    - z-score normalise per lead
    """
    signals, _ = wfdb.rdsamp(record_path)
    x = signals.astype(np.float32).T  # (12, T_raw)

    t = x.shape[1]
    if t > TARGET_LENGTH:
        start = (t - TARGET_LENGTH) // 2
        x = x[:, start:start + TARGET_LENGTH]
    elif t < TARGET_LENGTH:
        pad = TARGET_LENGTH - t
        x = np.pad(x, ((0, 0), (0, pad)), mode="constant")

    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    std[std < 1e-6] = 1.0
    x = (x - mean) / std

    return x.astype(np.float32)


def build_splits(df: pd.DataFrame, X: np.ndarray, y: np.ndarray):
    """
    Use PTB-XL recommended folds:
    - folds 1–8: training
    - fold 9   : validation
    - fold 10  : test
    """
    folds = df["strat_fold"].values.astype(int)

    train_mask = folds <= 8
    val_mask = folds == 9
    test_mask = folds == 10

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    return X_train, y_train, X_val, y_val, X_test, y_test


def print_class_distribution(name: str, y: np.ndarray, classes: np.ndarray) -> None:
    """Print class counts for a split."""
    print(f"\n{name} class distribution:")
    vals, counts = np.unique(y, return_counts=True)
    count_map = {int(v): int(c) for v, c in zip(vals, counts)}

    for idx, cls_name in enumerate(classes):
        count = count_map.get(idx, 0)
        print(f"  {cls_name}: {count}")


# --------- MAIN PIPELINE --------- #

def main():
    print("Loading metadata...")
    df, scp_df = load_metadata()
    scp_mapping = build_diagnostic_mapping(scp_df)

    if "sampling_frequency" in df.columns:
        df = df[df["sampling_frequency"] == TARGET_SAMPLING_RATE].copy()
        print(f"Total 500 Hz records (sampling_frequency == {TARGET_SAMPLING_RATE}): {len(df)}")
    else:
        print("Warning: 'sampling_frequency' column not found in ptbxl_database.csv")
        print("Using all records and loading waveforms via 'filename_hr'.")

    print(f"Label assignment mode: {'STRICT_SINGLE_SUPERCLASS' if STRICT_SINGLE_SUPERCLASS else 'HIGHEST_LIKELIHOOD'}")

    df["diagnostic_superclass"] = extract_superclass_labels(
        df,
        scp_mapping,
        strict_single_superclass=STRICT_SINGLE_SUPERCLASS,
    )
    df = df.dropna(subset=["diagnostic_superclass"]).reset_index(drop=True)
    print(f"Records with diagnostic superclass label: {len(df)}")

    X_list: List[np.ndarray] = []
    kept_rows: List[int] = []

    print("Loading and normalising ECG waveforms...")
    for idx, row in df.iterrows():
        record_path = os.path.join(BASE_PATH, row["filename_hr"])
        try:
            x = load_and_process_signal(record_path)
            X_list.append(x)
            kept_rows.append(idx)
        except Exception as e:
            print(f"  Skipping index {idx} ({record_path}): {e}")

    if len(X_list) == 0:
        raise RuntimeError("No ECG waveforms were successfully loaded.")

    if len(kept_rows) != len(df):
        dropped = len(df) - len(kept_rows)
        df = df.iloc[kept_rows].reset_index(drop=True)
        print(f"Dropped {dropped} records that could not be loaded.")

    X = np.stack(X_list, axis=0)  # (N, 12, TARGET_LENGTH)
    y_str = df["diagnostic_superclass"].values

    print("Encoding labels...")
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_str)

    X_train, y_train, X_val, y_val, X_test, y_test = build_splits(df, X, y)

    print("\nFinal shapes:")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val  : {X_val.shape}, y_val  : {y_val.shape}")
    print(f"  X_test : {X_test.shape}, y_test : {y_test.shape}")
    print(f"  Classes: {list(encoder.classes_)}")

    print_class_distribution("Train", y_train, encoder.classes_)
    print_class_distribution("Val", y_val, encoder.classes_)
    print_class_distribution("Test", y_test, encoder.classes_)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    np.savez_compressed(
        OUT_PATH,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        classes=encoder.classes_,
        strat_fold=df["strat_fold"].values.astype(np.int32),
        diagnostic_superclass=y_str,
        ecg_id=df["ecg_id"].values if "ecg_id" in df.columns else np.arange(len(df)),
    )
    print(f"\nSaved processed splits to: {OUT_PATH}")


if __name__ == "__main__":
    main()