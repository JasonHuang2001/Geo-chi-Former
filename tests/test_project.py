from __future__ import annotations

import gzip
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = ROOT / "checkpoints" / "paper_main"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


class ProjectTests(unittest.TestCase):
    def test_checkpoint_hash_strict_load_and_cpu_forward(self):
        from models.model_setup import apply_checkpoint_defaults, build_paper_model

        expected = {}
        for line in (CHECKPOINT_DIR / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
            checksum, name = line.split(maxsplit=1)
            expected[name] = checksum.upper()

        checkpoint = CHECKPOINT_DIR / "chiformer_forecasting_best.pth"
        config_path = CHECKPOINT_DIR / "config.json"
        self.assertEqual(sha256(checkpoint), expected[checkpoint.name])
        self.assertEqual(sha256(config_path), expected[config_path.name])

        config = SimpleNamespace(**json.loads(config_path.read_text(encoding="utf-8")))
        apply_checkpoint_defaults(config)
        model = build_paper_model(config, torch.device("cpu"))
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()

        generator = torch.Generator().manual_seed(20260826)
        history = torch.randn(1, config.seq_len, config.input_size, generator=generator)
        future_known = torch.randn(1, config.pred_len, len(config.known_slice), generator=generator)
        future_observed = torch.randn(1, config.pred_len, len(config.observed_slice), generator=generator)
        with torch.no_grad():
            output = model(history, future_known=future_known, future_observed=future_observed)
        self.assertEqual(tuple(output.shape), (1, config.pred_len, config.output_size))
        self.assertTrue(torch.isfinite(output).all())

    def test_wilson_round_trip(self):
        from utils.integrator import (
            WilsonPhysicsLayer,
            integrate_dataset_chi_with_wilson,
            invert_dataset_pole_with_wilson,
        )

        days = np.arange(0, 94, dtype=np.float64)
        x = 120.0 + 18.0 * np.sin(2.0 * np.pi * days / 37.0) + 0.04 * days
        y = -210.0 + 12.0 * np.cos(2.0 * np.pi * days / 29.0) - 0.03 * days
        pole = torch.tensor(np.column_stack([x[1:], y[1:]]), dtype=torch.float64).unsqueeze(0)
        pole_last = torch.tensor([[x[0], y[0]]], dtype=torch.float64)
        chi_last = torch.tensor([[8.0, -5.0]], dtype=torch.float64)
        physics = WilsonPhysicsLayer(dt=1.0, tc=433.0, q=179.0)
        chi = invert_dataset_pole_with_wilson(physics, pole, pole_last, chi_last)
        reconstructed, _ = integrate_dataset_chi_with_wilson(physics, chi, pole_last, chi_last)
        self.assertTrue(torch.allclose(reconstructed, pole, atol=1e-10, rtol=1e-10))

    def test_cli_help(self):
        for command in ("predict", "evaluate", "validate"):
            completed = subprocess.run(
                [sys.executable, "-B", str(ROOT / "run.py"), command, "--help"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, msg=completed.stderr)
            self.assertIn("usage:", completed.stdout)

    def test_plot_free_inference_writes_numeric_outputs(self):
        from utils.inference import evaluate_model_on_dataset

        class TinyDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 1

            def __getitem__(self, _index):
                history, horizon = 16, 3
                return (
                    torch.zeros(history, 2), torch.zeros(history, 8),
                    torch.zeros(horizon, 2), torch.zeros(horizon, 8),
                    torch.zeros(history, 2), torch.zeros(horizon, 2),
                    torch.zeros(history, 8), torch.zeros(horizon, 8),
                    torch.zeros(2), torch.zeros(2),
                    torch.zeros(history, 2), torch.zeros(horizon, 2),
                )

            def sample_future_dates(self, _index):
                return np.asarray(["2020-01-03", "2020-01-04", "2020-01-05"])

        class ZeroModel(torch.nn.Module):
            def forward(self, history, future_known=None, future_observed=None):
                return torch.zeros(history.shape[0], future_known.shape[1], 2, device=history.device)

        config = SimpleNamespace(
            batch_size=1, ul_gap_target_indices=[0, 1], use_future_observed_eam=True,
            dt=1.0, tc=433.0, q=179.0, metric_horizons=[1, 3],
        )
        with tempfile.TemporaryDirectory() as temporary:
            result = evaluate_model_on_dataset(
                ZeroModel(), TinyDataset(), config, torch.device("cpu"), "test", temporary
            )
            self.assertEqual(result["num_samples"], 1)
            self.assertTrue((Path(temporary) / "test_predictions.csv").is_file())
            metrics = json.loads((Path(temporary) / "test_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(sorted(metrics["horizons"]), ["1", "3"])

    def test_figure_data_and_previews(self):
        python_files = list(ROOT.rglob("*.py"))
        forbidden = ("matplotlib", ".plot(", ".savefig(", "plt.")
        for path in python_files:
            if path.resolve() == Path(__file__).resolve():
                continue
            text = path.read_text(encoding="utf-8")
            for marker in forbidden:
                self.assertNotIn(marker, text, msg=f"{marker!r} in {path.relative_to(ROOT)}")
        previews = sorted((ROOT / "figures" / "previews").glob("figure_*.png"))
        self.assertEqual(len(previews), 7)
        self.assertTrue(all(path.stat().st_size > 20_000 for path in previews))

        figure2 = ROOT / "plot" / "figure_02_distribution_shift"
        expected_rows = {
            "distribution.csv.gz": 126000,
            "gap_timeseries.csv.gz": 12052,
        }
        for name, rows in expected_rows.items():
            with gzip.open(figure2 / name, "rt", encoding="utf-8") as handle:
                self.assertEqual(sum(1 for _ in handle) - 1, rows)

    def test_restricted_prepared_inputs_are_absent(self):
        self.assertFalse((ROOT / "data" / "eop_data_xy_EAM.csv").exists())
        self.assertFalse((ROOT / "data" / "eam14forecast_daily.csv").exists())


if __name__ == "__main__":
    unittest.main()
