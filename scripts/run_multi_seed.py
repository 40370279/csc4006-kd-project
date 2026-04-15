import numpy as np

# Import training entry points for both models
from train_student_kd import main as run_kd
from train_student_baseline import main as run_baseline


# Seeds used for reproducibility and variance estimation
SEEDS = [42, 123, 999]


def run_experiment(run_fn, name):
    """
    Runs a given training function across multiple seeds
    and reports mean ± std for key metrics.

    Args:
        run_fn: function to execute (baseline or KD training)
        name: label for experiment (used in prints)
    """
    results = []

    print(f"\n===== RUNNING {name} =====")

    # Loop over predefined seeds
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")

        # Hack: override command-line args to inject seed
        # This ensures existing training scripts pick it up
        import sys
        sys.argv = [sys.argv[0], "--seed", str(seed)]

        # Run experiment and store result dictionary
        res = run_fn()
        results.append(res)

        # Print per-seed performance
        print(f"Seed {seed} -> acc={res['acc']:.4f}, macro_f1={res['macro_f1']:.4f}")

    # Extract metrics across all seeds
    accs = [r["acc"] for r in results]
    macros = [r["macro_f1"] for r in results]
    weights = [r["weighted_f1"] for r in results]

    # Print aggregated statistics
    print(f"\n===== {name} SUMMARY =====")
    print(f"Accuracy     : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"Macro F1     : {np.mean(macros):.4f} ± {np.std(macros):.4f}")
    print(f"Weighted F1  : {np.mean(weights):.4f} ± {np.std(weights):.4f}")


def main():
    """
    Runs both baseline and KD experiments sequentially.
    """
    run_experiment(run_baseline, "BASELINE")
    run_experiment(run_kd, "KD")


if __name__ == "__main__":
    # Entry point
    main()