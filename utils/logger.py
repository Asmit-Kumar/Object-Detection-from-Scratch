"""
RunLogger — Lightweight experiment tracker for object detection & character recognition.

Each run is saved as its own JSON file, organized by model name:

    logs/CharacterClassifier/
        characterclassifier_20260629_123045.json
        characterclassifier_20260629_145512.json   ← never overwritten

    logs/ObjectDetectorRes/
        objectdetectores_20260629_130000.json

Usage (via fit()):
    fit(..., log=True)   # log_dir is auto-derived from model class name

Usage (standalone):
    logger = RunLogger(log_dir="logs/MyModel")
    logger.start(config={"model": "CharacterClassifier", "lr": 1e-3})
    for epoch in range(epochs):
        logger.log_epoch(epoch, train_loss=tl, val_loss=vl, val_metric=acc, lr=lr)
    logger.finish()
    logger.summary()
"""

import json
import math
import time
from datetime import datetime
from pathlib import Path


class RunLogger:
    """
    Lightweight experiment logger — one JSON file per run, never overwritten.

    Args:
        run_name (str | None): Human-readable name for this run. When None
            (default), auto-generated in start() as {model}_{YYYYMMDD}_{HHMMSS}.
        log_dir (str): Directory in which run files are saved.
            Each run creates {log_dir}/{run_name}.json.
        verbose (bool): Print a one-line summary after each epoch.
        metric_unit (str): Unit label shown in summary tables (e.g. '%', 'IoU').
    """

    def __init__(
        self,
        run_name: str | None = None,
        log_dir: str = "logs/runs",
        verbose: int = 1,
        metric_unit: str = "%",
    ):
        self._run_name_override = run_name
        self.run_name = run_name or "(pending)"

        # Always anchor relative paths to the project root (parent of utils/)
        log_path = Path(log_dir)
        if not log_path.is_absolute():
            project_root = Path(__file__).resolve().parent.parent
            self.log_dir = project_root / log_path
        else:
            self.log_dir = log_path

        self.verbose = verbose
        self.metric_unit = metric_unit

        self._run: dict = {}
        self._epoch_metrics: list[dict] = []
        self._run_start: float = 0.0
        self._best_val_metric: float | None = None

        self.log_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self, config: dict | None = None) -> None:
        """
        Begin a new run. Call once before your training loop.

        Args:
            config: Arbitrary dict of hyperparameters to store alongside
                    the run (e.g. model name, lr, batch size, epochs…).
        """
        self._epoch_metrics = []
        self._run_start = time.time()
        self._best_val_metric = None
        cfg = config or {}

        if self._run_name_override is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_tag = cfg.get('label') if 'label' in cfg else cfg.get("model", "run")
            model_tag = model_tag.lower().replace(" ", "_").replace("/", "_")
            self.run_name = f"{model_tag}_{ts}"

        self._run = {
            "run_name":        self.run_name,
            "started_at":      datetime.now().isoformat(timespec="seconds"),
            "config":          cfg,
            "epochs":          [],
            "best_val_metric": None,
            "total_time_min":  None,
            "status":          "running",
        }

        if self.verbose >= 2 or self.verbose is True:
            print(f"[RunLogger] ▶  Run '{self.run_name}' started — "
                  f"saving to '{self._run_file}'")

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_metric: float,
        lr: float,
        epoch_time: float | None = None,
        metric_label: str = "val_metric",
        metric_unit: str | None = None,
        **extra,
    ) -> None:
        """
        Record metrics for a single epoch.

        Args:
            epoch:        0-based epoch index.
            train_loss:   Average training loss for this epoch.
            val_loss:     Average validation loss for this epoch.
            val_metric:   Primary validation metric (accuracy % or IoU).
            lr:           Current learning rate (last param group).
            epoch_time:   Wall-clock seconds for the epoch.
            metric_label: Display name for the metric column.
            metric_unit:  Unit string shown after the value.
            **extra:      Additional scalar metrics (e.g. val_mae=0.03, val_iou=0.85).
        """
        is_best = (
            self._best_val_metric is None
            or val_metric > self._best_val_metric
        )
        if is_best:
            self._best_val_metric = val_metric

        entry = {
            "epoch":      epoch + 1,
            "train_loss": round(train_loss, 6),
            "val_loss":   round(val_loss, 6),
            "val_metric": round(val_metric, 4),
            "lr":         round(lr, 8),
            "epoch_time": round(epoch_time, 2) if epoch_time is not None else None,
            "is_best":    is_best,
        }
        entry.update({k: round(v, 6) if isinstance(v, float) else v
                      for k, v in extra.items()})
        self._epoch_metrics.append(entry)

        if self.verbose >= 2 or self.verbose is True or (self.verbose == 1 and is_best):
            t_str = f"  ⏱ {epoch_time:.1f}s" if epoch_time else ""
            best_marker = "  ★ NEW BEST" if is_best else ""
            unit = metric_unit if metric_unit is not None else self.metric_unit
            print(
                f"[RunLogger] Epoch {epoch + 1:3d} | "
                f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"{metric_label}={val_metric:.4f}{unit}  "
                f"lr={lr:.2e}{t_str}{best_marker}"
            )

    def log_test_results(
        self,
        test_results: list,
        iou_threshold: float = 0.5,
        label: str = "threshold_sweep",
    ) -> None:
        """Persist test-set evaluation results into the run JSON.

        Note: `logger` is created internally by `fit()` and is not returned.
        To use this method, create a standalone RunLogger instance and call
        `logger.start()` / `logger.finish()` manually, or reload a run via
        `logger.load_run(run_name)`.

        Args:
            test_results:  List of dicts returned by evaluate_detection_sweep(),
                           each containing conf, precision, recall, f1, tp, fp, fn.
            iou_threshold: The IoU threshold used during the sweep.
            label:         Key name under which results are stored in the JSON.
                           Defaults to 'threshold_sweep'.

        Results are stored under run["test_results"][label] and written
        to disk immediately so they survive a crash.
        """
        if "test_results" not in self._run:
            self._run["test_results"] = {}
        self._run["test_results"][label] = {
            "iou_threshold": iou_threshold,
            "results": test_results,
        }
        self._save_run()

        if self.verbose >= 1:
            print(f"\n[RunLogger] Test sweep ({label}) — IoU threshold: {iou_threshold}")
            print(f"{'Conf':>6}  {'P':>6}  {'R':>6}  {'F1':>6}")
            print("-" * 32)
            for r in test_results:
                print(
                    f"{r['conf']:6.2f}  {r['precision']:6.4f}  "
                    f"{r['recall']:6.4f}  {r['f1']:6.4f}"
                )



    def finish(self, best_val_metric: float | None = None) -> None:
        """
        Finalise the run and write it to its own JSON file.

        Args:
            best_val_metric: Best validation metric from the run.
                             If None, computed automatically from logged epochs.
        """
        total_time = (time.time() - self._run_start) / 60.0

        if best_val_metric is None and self._epoch_metrics:
            best_val_metric = max(e["val_metric"] for e in self._epoch_metrics)

        if best_val_metric is None or not math.isfinite(float(best_val_metric)):
            best_val_metric = 0.0

        self._run.update({
            "epochs":          self._epoch_metrics,
            "best_val_metric": round(best_val_metric, 4),
            "total_time_min":  round(total_time, 2),
            "status":          "finished",
            "finished_at":     datetime.now().isoformat(timespec="seconds"),
        })

        self._save_run()

        if self.verbose >= 2 or self.verbose is True:
            print(
                f"[RunLogger] ■  Run '{self.run_name}' finished — "
                f"{len(self._epoch_metrics)} epochs in {total_time:.1f} min  |  "
                f"best val metric: {best_val_metric:.4f} {self.metric_unit}"
            )

    def summary(self, top_n: int = 10) -> None:
        """
        Print a ranked comparison table of all finished runs in log_dir.

        Args:
            top_n: Maximum number of runs to display (sorted by best val metric).
        """
        runs = self._load_all_runs()
        if not runs:
            print(f"[RunLogger] No runs found in '{self.log_dir}'.")
            return

        finished = [r for r in runs if r.get("status") == "finished"]
        finished.sort(key=lambda r: r.get("best_val_metric") or 0, reverse=True)
        display = finished[:top_n]

        col_w = [36, 14, 12, 8, 8]
        headers = ["Run Name", f"Best ({self.metric_unit})", "Time (min)", "Epochs", "LR"]
        sep = "-" * (sum(col_w) + len(headers) * 2)
        print(f"\n{'Run Summary':^{len(sep)}}")
        print(sep)
        print("  ".join(h.ljust(w) for h, w in zip(headers, col_w)))
        print(sep)

        for r in display:
            cfg = r.get("config", {})
            n_epochs = len(r.get("epochs", []))
            lr = cfg.get("lr", cfg.get("learning_rate", "—"))
            lr_str = f"{lr:.0e}" if isinstance(lr, float) else str(lr)
            vals = [
                r.get("run_name", "—")[:col_w[0]],
                f"{r.get('best_val_metric', 0):.4f} {self.metric_unit}",
                f"{r.get('total_time_min', 0):.1f}",
                str(n_epochs),
                lr_str,
            ]
            print("  ".join(v.ljust(w) for v, w in zip(vals, col_w)))

        print(sep)
        print(f"  Showing {len(display)} of {len(finished)} finished run(s)  |  "
              f"Dir: {self.log_dir}\n")

    def load_run(self, run_name: str) -> dict | None:
        """
        Load a specific run by name.

        Args:
            run_name: Run name to look up (matches the JSON filename stem).

        Returns:
            The run dict, or None if no matching file is found.
        """
        run_file = self.log_dir / f"{run_name}.json"
        if not run_file.exists():
            return None
        try:
            with open(run_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    def get_epoch_series(self, run_name: str | None = None) -> dict[str, list]:
        """
        Extract per-epoch metric lists from a run — ready for plotting.

        Args:
            run_name: Run to load. Defaults to the current in-progress run.

        Returns:
            Dict of lists keyed by metric name (e.g. 'train_loss', 'val_metric',
            'val_mae', 'val_iou'). All keys present in any epoch are included.
        """
        if run_name is None:
            epochs = self._epoch_metrics
        else:
            run = self.load_run(run_name)
            if run is None:
                raise KeyError(f"Run '{run_name}' not found in '{self.log_dir}'")
            epochs = run.get("epochs", [])

        if not epochs:
            return {}

        # Collect all scalar keys across all epoch entries
        all_keys: set[str] = set()
        for e in epochs:
            all_keys.update(e.keys())
        all_keys.discard("is_best")

        return {k: [e.get(k) for e in epochs] for k in sorted(all_keys)}

    # ── Internal helpers ──────────────────────────────────────────────────────

    @property
    def _run_file(self) -> Path:
        """Path to this run's dedicated JSON file."""
        return self.log_dir / f"{self.run_name}.json"

    def _load_all_runs(self) -> list[dict]:
        """Read every *.json file in log_dir and return as a list of run dicts."""
        if not self.log_dir.exists():
            return []
        runs = []
        for path in sorted(self.log_dir.glob("*.json")):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    runs.extend(data)
                else:
                    runs.append(data)
            except (json.JSONDecodeError, OSError):
                pass
        return runs

    def _save_run(self) -> None:
        """Write this run to its own dedicated JSON file."""
        with open(self._run_file, "w", encoding="utf-8") as f:
            json.dump(self._run, f, indent=2)
