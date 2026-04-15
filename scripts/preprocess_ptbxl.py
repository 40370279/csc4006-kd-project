import os
import ast
from typing import Tuple, Dict, List, Optional

import numpy as np
import pandas as pd
import wfdb
from sklearn.preprocessing import LabelEncoder


# --------- CONFIG --------- #

# Base directory for PTB-XL dataset
BASE_PATH = os.path.join("data", "ptbxl")

# Metadata and SCP statements files
DATABASE_CSV = os.path.join(BASE_PATH, "ptbxl_database.csv")
SCP_CSV = os.path.join(BASE_PATH, "scp_statements.csv")

# Target signal properties
TARGET_SAMPLING_RATE = 500          # Hz (high-resolution PTB-XL signals)
TARGET_LENGTH = 5000                # 10 seconds * 500 Hz

# Output file for processed dataset
OUT_PATH = os.path.join("processed", "ptbxl_500hz_10s.npz")

# Label assignment mode:
# True  -> keep only samples with a single consistent superclass
# False -> choose superclass with highest likelihood
STRICT_SINGLE_SUPERCLASS = True


# --------- HELPER FUNCTIONS --------- #

def load_metadata() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load PTB-XL metadata and SCP statements.

    Returns:
        df: main dataset metadata
        scp_df: SCP code definitions and mappings
    """
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

    Only includes diagnostic SCP codes with valid superclass labels.
    """
    # Filter to diagnostic codes only
    diag_df = scp_df[scp_df["diagnostic"] == 1].copy()

    # Remove rows without superclass labels
    diag_df = diag_df.dropna(subset=["diagnostic_class"])

    # Convert to dictionary: SCP code -> superclass
    mapping = diag_df["diagnostic_class"].to_dict()

    if not mapping:
        raise RuntimeError("Diagnostic mapping is empty – check scp_statements.csv format.")
    return mapping


def parse_scp_codes(scp_str: str) -> Dict[str, float]:
    """
    Safely parse SCP codes string into a dictionary.

    Example input:
        "{'AMI': 80, 'IMI': 50}"

    Returns:
        dict mapping code -> likelihood
    """
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
    Convert SCP codes for one ECG into a single superclass label.

    Modes:
    - strict_single_superclass=True:
        Keep only if all diagnostic codes map to SAME superclass
    - strict_single_superclass=False:
        Choose superclass of highest-likelihood SCP code
    """
    scp_dict = parse_scp_codes(scp_str)

    # Keep only codes that exist in mapping
    diag_items = [(code, likelihood) for code, likelihood in scp_dict.items() if code in scp_mapping]

    if not diag_items:
        return None

    # Non-strict: pick highest likelihood code
    if not strict_single_superclass:
        best_code, _ = max(diag_items, key=lambda t: t[1])
        return scp_mapping[best_code]

    # Strict: require all mapped codes to agree
    mapped_superclasses = {scp_mapping[code] for code, _ in diag_items}

    if len(mapped_superclasses) == 1:
        return next(iter(mapped_superclasses))

    return None  # conflicting labels → drop


def extract_superclass_labels(
    df: pd.DataFrame,
    scp_mapping: Dict[str, str],
    strict_single_superclass: bool = True,
) -> pd.Series:
    """
    Apply superclass mapping to entire dataset.

    Returns:
        Series of labels (may contain None for invalid rows)
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
    Load and preprocess a single ECG signal.

    Steps:
    - Load waveform using wfdb
    - Transpose to (leads, time)
    - Crop or pad to fixed length
    - Z-score normalise per lead
    """
    signals, _ = wfdb.rdsamp(record_path)

    # Convert to float32 and transpose (12, T)
    x = signals.astype(np.float32).T

    # Crop or pad to fixed length
    t = x.shape[1]
    if t > TARGET_LENGTH:
        start = (t - TARGET_LENGTH) // 2
        x = x[:, start:start + TARGET_LENGTH]
    elif t < TARGET_LENGTH:
        pad = TARGET_LENGTH - t
        x = np.pad(x, ((0, 0), (0, pad)), mode="constant")

    # Per-lead normalisation
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    std[std < 1e-6] = 1.0  # prevent division by zero

    x = (x - mean) / std

    return x.astype(np.float32)


def build_splits(df: pd.DataFrame, X: np.ndarray, y: np.ndarray):
    """
    Create train/val/test splits using PTB-XL folds:

    - folds 1–8 → training
    - fold 9    → validation
    - fold 10   → test
    """
    folds = df["strat_fold"].values.astype(int)

    train_mask = folds <= 8
    val_mask = folds == 9
    test_mask = folds == 10

    return (
        X[train_mask], y[train_mask],
        X[val_mask], y[val_mask],
        X[test_mask], y[test_mask],
    )


def print_class_distribution(name: str, y: np.ndarray, classes: np.ndarray) -> None:
    """
    Print class distribution for a dataset split.
    """
    print(f"\n{name} class distribution:")

    vals, counts = np.unique(y, return_counts=True)
    count_map = {int(v): int(c) for v, c in zip(vals, counts)}

    for idx, cls_name in enumerate(classes):
        count = count_map.get(idx, 0)
        print(f"  {cls_name}: {count}")


# --------- MAIN PIPELINE --------- #

def main():
    print("Loading metadata...")

    # Load dataset and mapping
    df, scp_df = load_metadata()
    scp_mapping = build_diagnostic_mapping(scp_df)

    # Filter to 500 Hz signals if available
    if "sampling_frequency" in df.columns:
        df = df[df["sampling_frequency"] == TARGET_SAMPLING_RATE].copy()
        print(f"Total 500 Hz records: {len(df)}")
    else:
        print("Warning: sampling_frequency not found — using all records.")

    print(f"Label mode: {'STRICT' if STRICT_SINGLE_SUPERCLASS else 'HIGHEST_LIKELIHOOD'}")

    # Assign superclass labels
    df["diagnostic_superclass"] = extract_superclass_labels(
        df,
        scp_mapping,
        strict_single_superclass=STRICT_SINGLE_SUPERCLASS,
    )

    # Drop invalid rows
    df = df.dropna(subset=["diagnostic_superclass"]).reset_index(drop=True)
    print(f"Valid labelled records: {len(df)}")

    # Load ECG signals
    X_list: List[np.ndarray] = []
    kept_rows: List[int] = []

    print("Loading ECG signals...")
    for idx, row in df.iterrows():
        record_path = os.path.join(BASE_PATH, row["filename_hr"])

        try:
            x = load_and_process_signal(record_path)
            X_list.append(x)
            kept_rows.append(idx)
        except Exception as e:
            print(f"Skipping {idx}: {e}")

    if len(X_list) == 0:
        raise RuntimeError("No signals loaded.")

    # Drop failed rows
    if len(kept_rows) != len(df):
        df = df.iloc[kept_rows].reset_index(drop=True)

    # Stack into array (N, 12, 5000)
    X = np.stack(X_list, axis=0)
    y_str = df["diagnostic_superclass"].values

    print("Encoding labels...")

    # Convert string labels → integers
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_str)

    # Create splits
    X_train, y_train, X_val, y_val, X_test, y_test = build_splits(df, X, y)

    # Print shapes
    print("\nFinal shapes:")
    print(f"X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")
    print(f"Classes: {list(encoder.classes_)}")

    # Print distributions
    print_class_distribution("Train", y_train, encoder.classes_)
    print_class_distribution("Val", y_val, encoder.classes_)
    print_class_distribution("Test", y_test, encoder.classes_)

    # Save dataset
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

    print(f"\nSaved to: {OUT_PATH}")


if __name__ == "__main__":
    main()