import numpy as np
from omega import compute_omega


def generate_synthetic_signals(n: int = 1000, seed: int = 42):
    np.random.seed(seed)
    t = np.arange(n)

    #  Pure sine wave
    pure_sine = np.sin(2 * np.pi * t / 50) + np.random.normal(0, 0.05, n)

    #  Daily weekly cycle
    dual_cycle = np.sin(2 * np.pi * t / 24) + np.sin(2 * np.pi * t / 168)

    # Moderately noisy signal
    noisy_signal = np.sin(2 * np.pi * t / 24) + np.random.normal(0, 0.5, n)

    # Pure white noise
    white_noise = np.random.normal(0, 1, n)

    return {
        "Pure sine wave": pure_sine,
        "Daily + weekly cycle": dual_cycle,
        "Moderately noisy signal": noisy_signal,
        "White noise": white_noise,
    }


def run_synthetic_tests():
    signals = generate_synthetic_signals()
    targets = {
        "Pure sine wave": 0.87,
        "Daily + weekly cycle": 0.79,
        "Moderately noisy signal": 0.32,
        "White noise": 0.06,
    }

    print("=" * 60)
    print("SYNTHETIC OMEGA TEST BATTERY VERIFICATION")
    print("=" * 60)
    print(f"{'Signal Type':<25} | {'Computed Omega':<15} | {'Plan Target':<12}")
    print("-" * 60)

    for name, signal in signals.items():
        arr = np.asarray(signal, dtype=np.float64)
        computed_val = float(compute_omega(arr, diffs=False))
        target_val = targets[name]
        print(f"{name:<25} | {computed_val:<15.4f} | ~{target_val:.2f}")

    print("=" * 60)


if __name__ == "__main__":
    run_synthetic_tests()