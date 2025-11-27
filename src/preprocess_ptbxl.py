#preprocess_ptbxl.py

import os
import ast
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import wfdb
from sklearn.preprocessing import LabelEncoder

# --------- CONFIG --------- #

# Folder where the PTB-XL archive was extracted
# e.g. data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3
BASE_PATH = os.path.join("data", "ptbxl")

DATABASE_CSV = os.path.join(BASE_PATH, "ptbxl_database.csv")
SCP_CSV = os.path.join(BASE_PATH, "scp_statements.csv")

TARGET_SAMPLING_RATE = 500          # Hz
TARGET_LENGTH = 5000                # 10 seconds * 500 Hz
OUT_PATH = os.path.join("processed", "ptbxl_500hz_10s.npz")


# --------- HELPER FUNCTIONS --------- #

def load_metadata() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load PTB-XL metadata and SCP statement information."""
    if not os.path.exists(DATABASE_CSV):
        raise FileNotFoundError(f"Could not find {DATABASE_CSV}")
    if not os.path.exists(SCP_CSV):
        raise FileNotFoundError(f"Could not find {SCP_CSV}")

    df = pd.read_csv(DATABASE_CSV)
    # Use SCP acronym as index – this matches the dict keys in scp_codes
    scp_df = pd.read_csv(SCP_CSV, index_col=0)

    return df, scp_df


def build_diagnostic_mapping(scp_df: pd.DataFrame) -> Dict[str, str]:
    """
    Build mapping from SCP code -> diagnostic superclass.

    We only keep statements that are marked as diagnostic and have a
    diagnostic_class such as NORM, MI, STTC, HYP or CD.
    """
    diag_df = scp_df[scp_df["diagnostic"] == 1]
    diag_df = diag_df.dropna(subset=["diagnostic_class"])

    mapping = diag_df["diagnostic_class"].to_dict()
    if not mapping:
        raise RuntimeError("Diagnostic mapping is empty – check scp_statements.csv format.")
    return mapping


def extract_superclass_labels(
    df: pd.DataFrame,
    scp_mapping: Dict[str, str],
) -> pd.Series:
    """
    Convert each row's scp_codes dict into a single diagnostic superclass label.

    Strategy:
    - Parse scp_codes (stringified dict: {code: likelihood})
    - Filter to diagnostic SCP codes that appear in scp_mapping
    - If multiple remain, take the one with highest likelihood
    """
    labels: List[str] = []

    for scp_str in df["scp_codes"]:
        # scp_codes stored as string representation of a dict
        scp_dict = ast.literal_eval(scp_str)  # e.g. {"NORM": 100, "IMI": 70}
        # keep only diagnostic codes we know
        diag_items = [(code, likelihood) for code, likelihood in scp_dict.items()
                      if code in scp_mapping]

        if not diag_items:
            labels.append(np.nan)
            continue

        # pick code with highest likelihood
        best_code, _ = max(diag_items, key=lambda t: t[1])
        labels.append(scp_mapping[best_code])

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

    # centre crop or right-pad with zeros to TARGET_LENGTH
    t = x.shape[1]
    if t > TARGET_LENGTH:
        start = (t - TARGET_LENGTH) // 2
        x = x[:, start:start + TARGET_LENGTH]
    elif t < TARGET_LENGTH:
        pad = TARGET_LENGTH - t
        x = np.pad(x, ((0, 0), (0, pad)), mode="constant")

    # per-lead normalisation
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    std[std < 1e-6] = 1.0  # avoid division by zero
    x = (x - mean) / std

    return x  # (12, TARGET_LENGTH)


def build_splits(df: pd.DataFrame, X: np.ndarray, y: np.ndarray):
    """
    Use PTB-XL's recommended stratified folds:
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


# --------- MAIN PIPELINE --------- #

def main():
    print("Loading metadata...")
    df, scp_df = load_metadata()
    scp_mapping = build_diagnostic_mapping(scp_df)

    # If 'sampling_frequency' is present, keep only 500 Hz records.
    # Otherwise, fall back to using all records and rely on filename_hr
    if "sampling_frequency" in df.columns:
        df = df[df["sampling_frequency"] == TARGET_SAMPLING_RATE].copy()
        print(f"Total 500 Hz records (sampling_frequency == {TARGET_SAMPLING_RATE}): {len(df)}")
    else:
        print("Warning: 'sampling_frequency' column not found in ptbxl_database.csv")
        print("→ Using all records and loading waveforms via 'filename_hr'.")
        # no filtering here; df stays as-is


    # build superclass labels
    df["diagnostic_superclass"] = extract_superclass_labels(df, scp_mapping)
    df = df.dropna(subset=["diagnostic_superclass"]).reset_index(drop=True)
    print(f"Records with diagnostic superclass label: {len(df)}")

    # load signals
    X_list: List[np.ndarray] = []
    bad_indices: List[int] = []

    print("Loading and normalising ECG waveforms...")
    for idx, row in df.iterrows():
        # PTB-XL gives 500 Hz files via `filename_hr`
        record_path = os.path.join(BASE_PATH, row["filename_hr"])
        try:
            X_list.append(load_and_process_signal(record_path))
        except Exception as e:
            print(f"  Skipping index {idx} ({record_path}): {e}")
            bad_indices.append(idx)

    if bad_indices:
        df = df.drop(index=bad_indices).reset_index(drop=True)
        print(f"Dropped {len(bad_indices)} records that could not be loaded.")

    X = np.stack(X_list, axis=0)  # (N, 12, TARGET_LENGTH)
    y_str = df["diagnostic_superclass"].values

    print("Encoding labels...")
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_str)

    # build splits using PTB-XL stratified folds
    X_train, y_train, X_val, y_val, X_test, y_test = build_splits(df, X, y)

    print("Final shapes:")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val  : {X_val.shape}, y_val  : {y_val.shape}")
    print(f"  X_test : {X_test.shape}, y_test : {y_test.shape}")
    print(f"  Classes: {list(encoder.classes_)}")

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
    )
    print(f"Saved processed splits to: {OUT_PATH}")


if __name__ == "__main__":
    main()
