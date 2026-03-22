import numpy as np

from train_student_kd import main as run_kd
from train_student_baseline import main as run_baseline


SEEDS = [42, 123, 999]


def run_experiment(run_fn, name):
    results = []

    print(f"\n===== RUNNING {name} =====")

    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")

        import sys
        sys.argv = [sys.argv[0], "--seed", str(seed)]

        res = run_fn()
        results.append(res)

        print(f"Seed {seed} -> acc={res['acc']:.4f}, macro_f1={res['macro_f1']:.4f}")

    accs = [r["acc"] for r in results]
    macros = [r["macro_f1"] for r in results]
    weights = [r["weighted_f1"] for r in results]

    print(f"\n===== {name} SUMMARY =====")
    print(f"Accuracy     : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"Macro F1     : {np.mean(macros):.4f} ± {np.std(macros):.4f}")
    print(f"Weighted F1  : {np.mean(weights):.4f} ± {np.std(weights):.4f}")


def main():
    run_experiment(run_baseline, "BASELINE")
    run_experiment(run_kd, "KD")


if __name__ == "__main__":
    main()