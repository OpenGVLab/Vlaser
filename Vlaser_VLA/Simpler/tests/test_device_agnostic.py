"""Device selection tests. The mocked ones need no accelerator; the parity test needs an XPU.

Run from Vlaser_VLA/Simpler: python -m pytest -q tests
"""

import glob
import sys
import tempfile
import types
import unittest
from unittest import mock

import torch
from omegaconf import OmegaConf

from src.utils import device as dev
from src.utils import monitor


def host(xpu=False, cuda=False):
    """Patch backend detection to fake a CPU-only, CUDA or XPU host."""
    return mock.patch.object(dev, "is_available", lambda t: {"xpu": xpu, "cuda": cuda}.get(t, t == "cpu"))


class DetectionTest(unittest.TestCase):
    def test_default_device_type_order(self):
        with host():
            self.assertEqual(dev.default_device_type(), "cpu")
        with host(cuda=True):
            self.assertEqual(dev.default_device_type(), "cuda")
        with host(xpu=True, cuda=True):
            self.assertEqual(dev.default_device_type(), "xpu")

    def test_resolve_none_uses_detected_backend_and_index(self):
        with host(xpu=True):
            self.assertEqual(dev.resolve_device(None), torch.device("xpu", 0))
            self.assertEqual(dev.resolve_device("", 1), torch.device("xpu", 1))
        with host(cuda=True):
            self.assertEqual(dev.resolve_device(None, 2), torch.device("cuda", 2))
        with host():
            self.assertEqual(dev.resolve_device(None), torch.device("cpu"))

    def test_explicit_device_wins(self):
        with host(xpu=True):
            self.assertEqual(dev.resolve_device("cuda:1"), torch.device("cuda", 1))
            self.assertEqual(dev.resolve_device("cpu", 3), torch.device("cpu"))
            self.assertEqual(dev.resolve_device("xpu:1", 0), torch.device("xpu", 1))
        with host(cuda=True):
            self.assertEqual(dev.resolve_device("xpu"), torch.device("xpu", 0))
            self.assertEqual(dev.resolve_device("cuda", 1), torch.device("cuda", 1))

    def test_eval_configs_detect_by_default(self):
        files = glob.glob("config/eval/*.yaml")
        self.assertTrue(files)
        for path in files:
            self.assertIsNone(OmegaConf.load(path).device, path)


class MonitorTest(unittest.TestCase):
    def test_memory_log_follows_device_type(self):
        lines = []
        with mock.patch.object(torch, "xpu", create=True) as xpu:
            xpu.memory_allocated.return_value = 1024**3
            monitor.log_allocated_gpu_memory(types.SimpleNamespace(info=lines.append), "x", torch.device("xpu", 0))
        self.assertEqual(len(lines), 1)
        self.assertIn("1.00 GB", lines[0])

    def test_memory_log_skips_cpu(self):
        lines = []
        monitor.log_allocated_gpu_memory(types.SimpleNamespace(info=lines.append), "x", torch.device("cpu"))
        self.assertEqual(lines, [])


class EvalAgentDeviceTest(unittest.TestCase):
    """EvalAgent moves the model to the resolved device (model, env and checkpoint mocked)."""

    def build(self, **cfg_kwargs):
        fake = {
            "simpler_env": mock.MagicMock(),
            "imageio": mock.MagicMock(),
            "src.agent.variant_aggressive_config": mock.MagicMock(),
            "src.model.vla.pizero_internvl": mock.MagicMock(),
        }
        with mock.patch.dict(sys.modules, fake), mock.patch.dict("os.environ", {"INTERNVL": "1"}):
            sys.modules.pop("src.agent.eval", None)
            from src.agent import eval as eval_mod

            cfg = OmegaConf.create(
                {
                    "n_eval_episode": 1,
                    "n_video": 0,
                    "log_dir": tempfile.mkdtemp(),
                    "gpu_id": 0,
                    "use_torch_compile": False,
                    "act_steps": 4,
                    "checkpoint_path": "unused.ckpt",
                    "env": {"task": "t", "adapter": {}},
                    **cfg_kwargs,
                }
            )
            with (
                mock.patch.object(eval_mod.hydra.utils, "instantiate"),
                mock.patch.object(eval_mod.EvalAgent, "load_checkpoint"),
            ):
                agent = eval_mod.EvalAgent(cfg)
            sys.modules.pop("src.agent.eval", None)
        return agent

    def test_detected_device_is_used(self):
        with host(xpu=True):
            agent = self.build(device=None)
        self.assertEqual(agent.device, torch.device("xpu", 0))
        agent.model.to.assert_called_with(torch.device("xpu", 0))

    def test_explicit_device_wins(self):
        with host(xpu=True):
            agent = self.build(device="cpu")
        self.assertEqual(agent.device, torch.device("cpu"))
        agent.model.to.assert_called_with(torch.device("cpu"))

    def test_cuda_host_keeps_cuda_index(self):
        with host(cuda=True):
            agent = self.build(device=None, gpu_id=1)
        self.assertEqual(agent.device, torch.device("cuda", 1))


def make_action_expert(dtype):
    """Action-expert pieces used in every flow-matching step, with every parameter re-randomised."""
    from src.model.vla.modules import ActionEncoder, AdaptiveRMSNorm, SinusoidalPosEmb

    torch.manual_seed(0)
    mods = torch.nn.ModuleDict(
        {"enc": ActionEncoder(7, 32, time_cond=True), "norm": AdaptiveRMSNorm(32, 32), "pos": SinusoidalPosEmb(32)}
    )
    for p in mods.parameters():
        torch.nn.init.normal_(p, std=0.3)
    return mods.to(dtype).eval()


def run_action_expert(mods, action, t):
    with torch.no_grad():
        cond = mods["pos"](t)
        return mods["norm"](mods["enc"](action, cond), cond)


@unittest.skipUnless(hasattr(torch, "xpu") and torch.xpu.is_available(), "needs an XPU")
class XpuParityTest(unittest.TestCase):
    def test_action_expert_cpu_vs_xpu(self):
        for dtype, tol in ((torch.float32, 1e-4), (torch.bfloat16, 5e-2)):
            mods = make_action_expert(dtype)
            action = torch.randn(2, 5, 7).to(dtype)  # noise is drawn on CPU, then moved
            t = torch.tensor([0.0, 0.5]).to(dtype)
            ref = run_action_expert(mods, action, t).float()
            self.assertGreater(ref.std().item(), 1e-2)  # not a constant output
            out = run_action_expert(mods.to("xpu"), action.to("xpu"), t.to("xpu")).float().cpu()
            self.assertTrue(torch.allclose(ref, out, atol=tol, rtol=tol), (dtype, (ref - out).abs().max()))


if __name__ == "__main__":
    unittest.main()
