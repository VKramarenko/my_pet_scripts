"""Example: load CSV, calibrate, print results.

Run:  python -m ToyOption.example
"""

from .data import CanonicalQuoteSet
from .service import ModelService


def main():
    # 1. Load data from CSV
    qs = CanonicalQuoteSet.from_csv("ToyOption/example_data.csv")
    print(f"Loaded: F={qs.F}, T={qs.T}, {len(qs.calls)} calls, {len(qs.puts)} puts\n")

    # 2. Set up service and calibrate
    svc = ModelService()
    svc.set_data(qs)
    result = svc.calibrate()

    # 3. Print calibration results
    params = dict(zip(svc.model.param_names, result.params))
    print(f"Model: {svc.model.name}")
    print(f"Calibration: {'OK' if result.success else 'FAILED'}")
    print(f"Parameters:")
    for name, val in params.items():
        print(f"  {name} = {val:.4f}")
    print(f"\nRMSE = {result.metrics['rmse']:.4f}")
    print(f"MAE  = {result.metrics['mae']:.4f}")
    print(f"Max  = {result.metrics['max_error']:.4f}")

    # 4. No-arbitrage diagnostics
    diag = svc.get_diagnostics()
    print("\nNo-arb checks:")
    for chk in diag["noarb"]:
        icon = "OK" if chk["ok"] else "FAIL"
        print(f"  [{icon}] {chk['name']}: {chk['detail']}")

    # 5. Show model vs market on input strikes
    print("\n--- Call prices ---")
    print(f"{'K':>8} {'Market':>8} {'Model':>8} {'Err':>8}")
    for K, mkt, _ in qs.calls:
        mdl = svc.model.price("call", K, qs.F, qs.T, svc.params)
        print(f"{K:8.1f} {mkt:8.2f} {mdl:8.2f} {mdl - mkt:+8.2f}")

    print("\n--- Put prices ---")
    print(f"{'K':>8} {'Market':>8} {'Model':>8} {'Err':>8}")
    for K, mkt, _ in qs.puts:
        mdl = svc.model.price("put", K, qs.F, qs.T, svc.params)
        print(f"{K:8.1f} {mkt:8.2f} {mdl:8.2f} {mdl - mkt:+8.2f}")


if __name__ == "__main__":
    main()
