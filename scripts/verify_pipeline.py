"""
scripts/verify_pipeline.py — Verification and Benchmark Script for the End-to-End Pipeline.

Tests:
  1. Single image prediction & visualization smoke test
  2. Batch GPU parallel inference throughput (FPS benchmark)
  3. Multi-Stream CUDA parallel evaluation across all 4 placement benchmarks (random, grid, words, line)
"""

import sys, time
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / 'generator'))
sys.path.insert(0, str(root_dir))

import torch
from utils.pipeline import DetectionPipeline
from utils.dataset import get_detection_loaders, ROOT_DIR


def main():
    print("=" * 65)
    print("  End-to-End Scene Understanding Pipeline Smoke Test & Benchmark")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Instantiate pipeline with Small Detector + ByMerge Classifier
    pipeline = DetectionPipeline(
        detector_size="s",
        detector_weights="weights/detector_s_new_best.pth",
        classifier_weights="weights/classifier_resent_bymerge_s_best.pth",
        device=device,
        conf_threshold=0.70,
    )
    print("\n[OK] Pipeline successfully initialized!")

    # 1. Single Image Prediction Test
    test_img_dir = Path("data/OD_benchmark/words/test/images")
    if not test_img_dir.exists():
        test_img_dir = ROOT_DIR / "test" / "images"

    sample_imgs = list(test_img_dir.glob("*.png"))
    if sample_imgs:
        sample_path = sample_imgs[0]
        print(f"\nRunning single-image prediction on '{sample_path.name}'...")
        results = pipeline.predict_image(sample_path)
        print(f"Detected {len(results)} characters:")
        for r in results[:10]:
            print(f"  bbox={r.bbox} | char='{r.char_label}' | det_conf={r.detector_conf:.4f} | cls_conf={r.classifier_conf:.4f} | joint_conf={r.joint_conf:.4f}")

    # 2. Batch Inference Throughput Test
    print("\n--- Running Batch Inference Throughput Benchmark ---")
    test_loader = get_detection_loaders(
        data_root=ROOT_DIR,
        batch_size=64,
        test_only=True,
        num_workers=0,
    )

    batch = next(iter(test_loader))
    images_batch = batch[0].to(device)

    # Warmup
    _ = pipeline.predict_batch(images_batch)

    # Benchmark 20 iterations
    t0 = time.time()
    total_imgs = 0
    N_ITERS = 20
    for _ in range(N_ITERS):
        batch_res = pipeline.predict_batch(images_batch)
        total_imgs += len(batch_res)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.time() - t0
    fps = total_imgs / elapsed
    print(f"Processed {total_imgs} images ({N_ITERS} batches of 64) in {elapsed:.2f}s -> {fps:.1f} Images/sec (FPS)")

    # 3. Parallel 4-Way Multi-Stream Benchmark Evaluation
    benchmark_dir = Path("data/OD_benchmark")
    if benchmark_dir.exists():
        print(f"\n" + "=" * 65)
        print("  Running Multi-Stream Parallel Evaluation on All 4 Benchmarks")
        print("=" * 65)
        metrics = pipeline.evaluate_benchmark_parallel(benchmark_dir=benchmark_dir, batch_size=128)

        print("\n" + "=" * 70)
        print(f"  {'Layout':<10} | {'Det Precision':<13} | {'Det Recall':<10} | {'Cls Acc':<9} | {'E2E F1':<8} | {'FPS':<6}")
        print("=" * 70)
        for layout, m in metrics.items():
            if "error" in m:
                print(f"  {layout:<10} | ERROR: {m['error']}")
            else:
                print(
                    f"  {layout:<10} | "
                    f"{m['det_precision']:<13.4f} | "
                    f"{m['det_recall']:<10.4f} | "
                    f"{m['classifier_acc']:<9.4f} | "
                    f"{m['e2e_f1']:<8.4f} | "
                    f"{m['fps']:<6.1f}"
                )
        print("=" * 70)

    print("\n[SUCCESS] Pipeline verification complete!")


if __name__ == "__main__":
    main()
