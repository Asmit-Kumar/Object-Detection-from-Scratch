"""Extract key metrics from the three model run logs for the INSIGHTS/README update."""
import json, sys, os
sys.stdout.reconfigure(encoding='utf-8')

LOG_DIR = "logs/ObjectDetectorResNet"

# The three comparison runs
runs = {
    "n": "detector_n_v2_20260718_121834.json",
    "s": "detector_s_v2_20260719_100048.json",
    "m": "detector_m_20260718_144358.json",
}

for size, fname in runs.items():
    path = os.path.join(LOG_DIR, fname)
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    epochs = data.get("epochs", [])
    best_iou = max(e["val_metric"] for e in epochs)
    best_epoch = next(e for e in epochs if e["val_metric"] == best_iou)

    # test results if present
    test = data.get("test_results", {})
    sweep = test.get("threshold_sweep", {}).get("results", [])

    # best F1 from sweep
    best_f1_row = max(sweep, key=lambda r: r["f1"]) if sweep else None

    print(f"\n{'='*55}")
    print(f"  Size: {size.upper()}  |  {fname}")
    print(f"{'='*55}")
    print(f"  Total epochs   : {len(epochs)}")
    print(f"  Best val IoU   : {best_iou:.4f}")
    print(f"  Final train loss: {epochs[-1]['train_loss']:.4f}")
    print(f"  Final val loss  : {epochs[-1]['val_loss']:.4f}")
    if best_f1_row:
        print(f"  Sweep best F1  : {best_f1_row['f1']:.4f} @ conf={best_f1_row['conf']}")
        print(f"    P={best_f1_row['precision']:.4f}  R={best_f1_row['recall']:.4f}")
        print(f"    TP={best_f1_row['tp']}  FP={best_f1_row['fp']}  FN={best_f1_row['fn']}")
    if sweep:
        print(f"\n  Full sweep:")
        print(f"  {'Conf':>6}  {'P':>6}  {'R':>6}  {'F1':>6}")
        for r in sweep:
            print(f"  {r['conf']:6.2f}  {r['precision']:6.4f}  {r['recall']:6.4f}  {r['f1']:6.4f}")
