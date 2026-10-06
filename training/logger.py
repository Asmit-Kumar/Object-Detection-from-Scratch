"""
RunLogger — Lightweight experiment tracker for object detection & character recognition.

Each run is saved as its own JSON file, organized by model name:

    logs/CharacterClassifier/
        characterclassifier_20260629_123045.json
        characterclassifier_20260629_145512.json   ← never overwritten

    logs/ObjectDetectorRes/
        objectdetectores_20260629_130000.json

Persistence guarantees:
  * The run file is written at start() and after every logged epoch (atomic
    write-then-rename), so a crash mid-training keeps every finished epoch.
  * Separate runs always get separate files: start() without resume (or with no
    matching checkpoint) creates a new timestamped file.
  * start(resume=True) re-opens the most recent run of the same model/label and
    appends to it, but only if it is consistent with the checkpoint being
    resumed; otherwise a new file is created.
  * Epoch entries are append-only. If an epoch number is logged again (e.g. the
    run resumed from a checkpoint older than the last logged epoch) the earlier
    entry is kept and the new one is stored with attempt=2, 3, ...
  * Test-set results and any other post-training output (final evaluation
    metrics, threshold sweeps, notes) are stored in the same run file. Logging
    the same label again archives the previous value instead of replacing it.

Usage (via fit()):
    fit(..., log=True)   # log_dir is auto-derived from model class name
    history, logger = fit(...)
    logger.log_test_results(sweep_results, iou_threshold=0.5)
    logger.log_test_metrics(metrics, conf_threshold=0.5)

Usage (standalone):
    logger = RunLogger(log_dir="logs/MyModel")
    logger.start(config={"model": "CharacterClassifier", "lr": 1e-3})
    for epoch in range(epochs):
        logger.log_epoch(epoch, train_loss=tl, val_loss=vl, val_metric=acc, lr=lr)
    logger.finish()
    logger.summary()

Attach to an already finished run (e.g. new notebook session, test-time logging):
    logger = RunLogger(log_dir="logs/MyModel")
    logger.attach()                       # latest run, or attach("run_name")
    logger.log_test_results(sweep)
"""

import json
import math
import os
import re
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
        self._prior_time_min: float = 0.0
        self._track_time: bool = True
        self._best_val_metric: float | None = None

        self.log_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def start(
        self,
        config: dict | None = None,
        resume: bool = False,
        start_epoch: int = 0,
    ) -> None:
        """
        Begin a run. Call once before your training loop.

        Args:
            config: Arbitrary dict of hyperparameters to store alongside
                    the run (e.g. model name, lr, batch size, epochs…).
            resume: If True, continue the most recent run of the same
                    model/label found in log_dir (all earlier epochs, test
                    results and timing are kept and new epochs are appended).
                    Falls back to a fresh run when none exists.
            start_epoch: 0-based index of the first epoch that will be trained
                    on resume. Used only to record which logged epochs are
                    being re-run; nothing is deleted or overwritten.
        """
        self._epoch_metrics = []
        self._run_start = time.time()
        self._prior_time_min = 0.0
        self._track_time = True
        self._best_val_metric = None
        cfg = config or {}

        model_tag = cfg.get('label') if 'label' in cfg else cfg.get("model", "run")
        model_tag = model_tag.lower().replace(" ", "_").replace("/", "_")

        if resume:
            existing = self._find_resumable_run(model_tag, start_epoch)
            if existing is not None:
                self._adopt_run(existing, cfg, start_epoch)
                return

        if self._run_name_override is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_name = f"{model_tag}_{ts}"

        self._run = {
            "run_name":        self.run_name,
            "started_at":      datetime.now().isoformat(timespec="seconds"),
            "config":          cfg,
            "epochs":          self._epoch_metrics,
            "best_val_metric": None,
            "total_time_min":  0.0,
            "status":          "running",
        }
        self._save_run()

        if self.verbose >= 2 or self.verbose is True:
            print(f"[RunLogger] ▶  Run '{self.run_name}' started — "
                  f"saving to '{self._run_file}'")

    def attach(self, run_name: str | None = None) -> bool:
        """
        Attach to an existing run file without starting a new one, so that
        test results / extras can be added after training (e.g. from a fresh
        notebook session).

        Args:
            run_name: Run to attach to. Defaults to the most recent run in log_dir.

        Returns:
            True if a run was attached, False if none was found.
        """
        if run_name is None:
            run_name = self._latest_run_name()
        run = self.load_run(run_name) if run_name else None
        if run is None:
            return False
        self.run_name = run["run_name"]
        self._run_name_override = self.run_name
        self._run = run
        self._epoch_metrics = run.setdefault("epochs", [])
        self._prior_time_min = float(run.get("total_time_min") or 0.0)
        self._run_start = time.time()
        self._track_time = False        # attaching only adds results; idle time is not training time
        self._best_val_metric = self._best_logged_metric()
        return True

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
        Record metrics for a single epoch and persist them to disk immediately.

        Epoch entries are append-only: logging an epoch number that already
        exists keeps the earlier entry and stores this one with attempt=2, 3, ...

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

        previous = [e.get("attempt", 1) for e in self._epoch_metrics
                    if e.get("epoch") == entry["epoch"]]
        if previous:
            entry["attempt"] = max(previous) + 1

        self._epoch_metrics.append(entry)
        self._save_run()

        if self.verbose >= 2 or self.verbose is True or (self.verbose == 1 and is_best):
            t_str = f"  ⏱ {epoch_time:.1f}s" if epoch_time else ""
            best_marker = "  ★ NEW BEST" if is_best else ""
            unit = metric_unit if metric_unit is not None else self.metric_unit
            attempt_str = f" (attempt {entry['attempt']})" if "attempt" in entry else ""
            print(
                f"[RunLogger] Epoch {epoch + 1:3d}{attempt_str} | "
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

        `fit()` returns `(history, logger)`, so the returned logger can be used
        directly after training. In a new session use `logger.attach()` first.

        Args:
            test_results:  List of dicts from a confidence-threshold sweep,
                           each containing conf, precision, recall, f1, tp, fp, fn
                           (plus any extra keys, e.g. cls_accuracy, per_class_e2e).
            iou_threshold: The IoU threshold used during the sweep.
            label:         Key name under which results are stored in the JSON.
                           Defaults to 'threshold_sweep'.

        Results are stored under run["test_results"][label] and written
        to disk immediately so they survive a crash. Logging the same label
        again keeps the previous value under run["test_results_history"][label].
        """
        self._store("test_results", label, {
            "iou_threshold": iou_threshold,
            "results": test_results,
        })

        if self.verbose >= 1:
            print(f"\n[RunLogger] Test sweep ({label}) — IoU threshold: {iou_threshold}")
            print(f"{'Conf':>6}  {'P':>6}  {'R':>6}  {'F1':>6}")
            print("-" * 32)
            for r in test_results:
                print(
                    f"{r['conf']:6.2f}  {r['precision']:6.4f}  "
                    f"{r['recall']:6.4f}  {r['f1']:6.4f}"
                )

    def log_test_metrics(
        self,
        metrics: dict,
        conf_threshold: float | None = None,
        iou_threshold: float = 0.5,
        label: str = "final_eval",
    ) -> None:
        """Persist the full dict returned by `evaluate_detection()` (loss,
        detection, classification incl. per-class/confusion matrix, end_to_end).

        Args:
            metrics:        Output of evaluate_detection(...).
            conf_threshold: Confidence threshold the evaluation used.
            iou_threshold:  IoU threshold the evaluation used.
            label:          Key under run["test_results"]. Defaults to 'final_eval'.
        """
        self._store("test_results", label, {
            "conf_threshold": conf_threshold,
            "iou_threshold": iou_threshold,
            "metrics": metrics,
        })

        if self.verbose >= 1:
            det = metrics.get("detection", {})
            e2e = metrics.get("end_to_end", {})
            cls = metrics.get("classification", {})
            print(
                f"\n[RunLogger] Test metrics ({label}) saved - "
                f"Det P={det.get('precision', 0):.4f} R={det.get('recall', 0):.4f} "
                f"F1={det.get('f1', 0):.4f} | Cls acc={cls.get('accuracy', 0):.4f} | "
                f"E2E F1={e2e.get('f1', 0):.4f}"
            )

    def log_extra(self, label: str, data) -> None:
        """Persist any JSON-serialisable value (best threshold, notes, per-layout
        results…) under run["extras"][label]. Same-label values are archived,
        not replaced."""
        self._store("extras", label, data)

    def finish(self, best_val_metric: float | None = None) -> None:
        """
        Finalise the run and write it to its own JSON file.

        Args:
            best_val_metric: Best validation metric from the run.
                             If None, computed automatically from logged epochs.
        """
        total_time = self._total_time_min()

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
                f"{len({e['epoch'] for e in self._epoch_metrics})} epochs in {total_time:.1f} min  |  "
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
            n_epochs = len({e.get("epoch") for e in r.get("epochs", [])})
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

    def get_epoch_series(
        self,
        run_name: str | None = None,
        include_superseded: bool = False,
    ) -> dict[str, list]:
        """
        Extract per-epoch metric lists from a run — ready for plotting.

        Args:
            run_name: Run to load. Defaults to the current in-progress run.
            include_superseded: If False (default) only the latest attempt of
                each epoch is returned (one point per epoch, in epoch order).
                If True every logged entry is returned in logging order.

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

        if not include_superseded:
            latest: dict = {}
            for e in epochs:
                latest[e.get("epoch")] = e      # later attempts are appended later
            epochs = [latest[k] for k in sorted(latest)]

        # Collect all scalar keys across all epoch entries
        all_keys: set[str] = set()
        for e in epochs:
            all_keys.update(e.keys())
        all_keys.discard("is_best")
        if not include_superseded:
            all_keys.discard("attempt")

        return {k: [e.get(k) for e in epochs] for k in sorted(all_keys)}

    # ── Internal helpers ──────────────────────────────────────────────────────

    @property
    def _run_file(self) -> Path:
        """Path to this run's dedicated JSON file."""
        return self.log_dir / f"{self.run_name}.json"

    def _total_time_min(self) -> float:
        """Training minutes across all sessions of this run."""
        return self._prior_time_min + (time.time() - self._run_start) / 60.0

    def _best_logged_metric(self) -> float | None:
        values = [e["val_metric"] for e in self._epoch_metrics if "val_metric" in e]
        return max(values) if values else None

    def _matching_run_names(self, model_tag: str | None = None) -> list[str]:
        """Runs in log_dir (optionally restricted to `<tag>_YYYYMMDD_HHMMSS`),
        newest first by start timestamp."""
        if self._run_name_override is not None and model_tag is not None:
            name = self._run_name_override
            return [name] if self._run_file_for(name).exists() else []
        pattern = (
            re.compile(rf"^{re.escape(model_tag)}_(\d{{8}}_\d{{6}})$")
            if model_tag is not None else re.compile(r"^.+_(\d{8}_\d{6})$")
        )
        candidates = []
        for path in self.log_dir.glob("*.json"):
            m = pattern.match(path.stem)
            if m:
                candidates.append((m.group(1), path.stem))
        return [name for _, name in sorted(candidates, reverse=True)]

    def _latest_run_name(self, model_tag: str | None = None) -> str | None:
        """Newest run in log_dir (optionally restricted to a model tag)."""
        names = self._matching_run_names(model_tag)
        return names[0] if names else None

    def _run_file_for(self, run_name: str) -> Path:
        return self.log_dir / f"{run_name}.json"

    def _find_resumable_run(self, model_tag: str, start_epoch: int) -> dict | None:
        """Newest run of this label, but only if it is a plausible continuation of
        the checkpoint being resumed (otherwise a new run file is started, so
        separate runs never get merged into one log)."""
        names = self._matching_run_names(model_tag)
        skipped = []
        for run_name in names:                      # newest first
            run = self.load_run(run_name)
            if run is None:
                continue
            logged = [e.get("epoch", 0) for e in run.get("epochs", [])]
            last = max(logged) if logged else 0
            if last < start_epoch - 1:
                reason = f"only {last} epoch(s) logged, checkpoint is at epoch {start_epoch}"
            elif run.get("status") == "finished" and last > start_epoch:
                reason = f"finished at epoch {last}, ahead of the checkpoint (epoch {start_epoch})"
            else:
                if skipped and self.verbose >= 1:
                    print(f"[RunLogger] Skipped {len(skipped)} newer run(s) that do not match the "
                          f"checkpoint (epoch {start_epoch}): {', '.join(skipped)}")
                return run
            skipped.append(f"{run_name} ({reason})")

        if names and self.verbose >= 1:
            print(f"[RunLogger] No existing run matches the checkpoint (epoch {start_epoch}); "
                  f"starting a new log file. Checked: {', '.join(skipped)}")
        return None

    def _adopt_run(self, run: dict, cfg: dict, start_epoch: int) -> None:
        """Continue an existing run: keep everything on disk and append to it."""
        self.run_name = run["run_name"]
        self._run = run
        self._epoch_metrics = run.setdefault("epochs", [])
        self._prior_time_min = float(run.get("total_time_min") or 0.0)
        self._best_val_metric = self._best_logged_metric()

        next_epoch = start_epoch + 1          # 1-based, as stored in the log
        rerun = sorted({e["epoch"] for e in self._epoch_metrics if e.get("epoch", 0) >= next_epoch})
        old_cfg = run.get("config", {})
        self._run.setdefault("resumes", []).append({
            "resumed_at":          datetime.now().isoformat(timespec="seconds"),
            "next_epoch":          next_epoch,
            "previous_status":     run.get("status"),
            "epochs_on_disk":      len({e.get("epoch") for e in self._epoch_metrics}),
            "epochs_being_rerun":  rerun,
            "config_changes":      {k: v for k, v in cfg.items() if old_cfg.get(k) != v},
        })
        self._run["status"] = "running"
        self._run.pop("finished_at", None)
        self._save_run()

        if self.verbose >= 1:
            note = f", re-running epochs {rerun[0]}-{rerun[-1]} as new attempts" if rerun else ""
            print(f"[RunLogger] Resuming run '{self.run_name}' - "
                  f"{len(self._epoch_metrics)} logged epoch(s) kept, next epoch {next_epoch}{note}")

    def _store(self, section: str, label: str, payload) -> None:
        """Save `payload` at run[section][label]; archive any previous value
        under run[section + '_history'][label] instead of overwriting it."""
        entry = {"logged_at": datetime.now().isoformat(timespec="seconds")}
        if isinstance(payload, dict):
            entry.update(payload)
        else:
            entry["value"] = payload
        bucket = self._run.setdefault(section, {})
        if label in bucket:
            self._run.setdefault(f"{section}_history", {}).setdefault(label, []).append(bucket[label])
        bucket[label] = entry
        self._save_run()

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
        """Write this run to its own dedicated JSON file (atomic: temp file + rename)."""
        if not self._run:
            return

        def _default(obj):
            if hasattr(obj, "tolist"):
                return obj.tolist()
            if hasattr(obj, "item"):
                return obj.item()
            return str(obj)

        self._run["epochs"] = self._epoch_metrics
        if self._track_time and self._run.get("status") != "finished":
            self._run["total_time_min"] = round(self._total_time_min(), 2)

        tmp_file = self._run_file.with_suffix(".json.tmp")
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(self._run, f, indent=2, default=_default)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_file, self._run_file)
