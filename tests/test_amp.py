"""
The trainer picks its autocast dtype per GPU: bfloat16 only where the GPU runs it natively
(compute capability 8.0+), float16 with a GradScaler on older GPUs such as the Kaggle T4.
CUDA is mocked, so these run on CPU-only machines.
"""
import ast
import inspect
import unittest
import warnings
from unittest import mock

import torch

from training import trainer

CUDA = torch.device("cuda")
CPU = torch.device("cpu")


def on_gpu(capability):
    """Pretend a CUDA GPU with the given compute capability is present."""
    return mock.patch.multiple(
        torch.cuda,
        is_available=mock.Mock(return_value=True),
        get_device_capability=mock.Mock(return_value=capability),
    )


class TestAmpDtype(unittest.TestCase):

    def test_native_bf16_gpus_keep_bfloat16(self):
        for capability in ((8, 0), (8, 6), (8, 9), (9, 0), (12, 0)):  # A100, RTX 30xx, RTX 40xx, H100, RTX 50xx
            with self.subTest(capability=capability), on_gpu(capability):
                self.assertIs(trainer._amp_dtype(CUDA), torch.bfloat16)

    def test_older_gpus_use_float16(self):
        for capability in ((7, 5), (7, 0), (6, 0)):  # T4, V100, P100
            with self.subTest(capability=capability), on_gpu(capability):
                self.assertIs(trainer._amp_dtype(CUDA), torch.float16)

    def test_cpu_is_unchanged(self):
        self.assertIs(trainer._amp_dtype(CPU), torch.bfloat16)


class TestGradScalerResolution(unittest.TestCase):

    def resolve(self, scaler, device):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # GradScaler warns that CUDA is absent on CPU-only machines
            return trainer._resolve_grad_scaler(scaler, device)

    def test_float16_without_a_scaler_gets_one(self):
        with on_gpu((7, 5)):
            self.assertIsInstance(self.resolve(None, CUDA), torch.amp.GradScaler)

    def test_bfloat16_stays_unscaled(self):
        with on_gpu((8, 6)):
            self.assertIsNone(self.resolve(None, CUDA))

    def test_a_given_scaler_is_kept(self):
        given = object()
        for capability in ((7, 5), (8, 6)):
            with self.subTest(capability=capability), on_gpu(capability):
                self.assertIs(self.resolve(given, CUDA), given)

    def test_cpu_needs_no_scaler(self):
        self.assertIsNone(self.resolve(None, CPU))


class TestNoHardCodedDtype(unittest.TestCase):

    def test_every_autocast_uses_the_per_device_dtype(self):
        tree = ast.parse(inspect.getsource(trainer))
        calls = [node for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "autocast"]
        self.assertGreater(len(calls), 0)
        for call in calls:
            dtype = next(kw.value for kw in call.keywords if kw.arg == "dtype")
            self.assertEqual(ast.unparse(dtype), "_amp_dtype(device)", ast.unparse(call))

    def test_every_function_using_it_has_a_device(self):
        tree = ast.parse(inspect.getsource(trainer))
        for fn in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)):
            if "_amp_dtype(device)" in ast.unparse(fn) and fn.name != "_resolve_grad_scaler":
                self.assertIn("device", [a.arg for a in fn.args.args], fn.name)


if __name__ == "__main__":
    unittest.main()
