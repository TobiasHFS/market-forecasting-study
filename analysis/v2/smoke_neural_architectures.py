"""Small deterministic architecture contract test for the v2 neural models."""

from __future__ import annotations

import json
import sys
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
for path in (
    WORKSPACE_ROOT / ".analysis_deps",
    WORKSPACE_ROOT / "analysis",
    WORKSPACE_ROOT / "analysis" / "v2",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import torch

from run_realmlp_challenger import NeuralSpec, _build_model, _set_determinism
from run_tabm_mini_challenger import TabMMiniSpec, _build_tabm_mini


def main() -> None:
    _set_determinism(torch, 20260823)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(32, 19, device=device)
    y = torch.randn(32, device=device)

    realmlp_spec = NeuralSpec(
        hidden_size=32,
        hidden_layers=3,
        max_epochs=2,
        min_epochs=1,
        patience=1,
    )
    realmlp = _build_model(torch, 19, realmlp_spec).to(device)
    realmlp_prediction = realmlp(x).squeeze(-1)
    if realmlp_prediction.shape != (32,):
        raise AssertionError("RealMLP output contract failed")
    realmlp_loss = torch.square(realmlp_prediction - y).mean()
    realmlp_loss.backward()

    tabm_spec = TabMMiniSpec(
        k=16,
        hidden_size=32,
        hidden_layers=2,
        max_epochs=2,
        min_epochs=1,
        patience=1,
    )
    tabm = _build_tabm_mini(torch, 19, tabm_spec).to(device)
    member_prediction = tabm(x)
    if member_prediction.shape != (32, 16):
        raise AssertionError("TabM member-output contract failed")
    individual_member_loss = torch.square(
        member_prediction - y.unsqueeze(1)
    ).mean()
    loss_of_mean_prediction = torch.square(member_prediction.mean(dim=1) - y).mean()
    if torch.isclose(individual_member_loss, loss_of_mean_prediction):
        raise AssertionError("TabM loss contract test failed to distinguish objectives")
    individual_member_loss.backward()

    for name, model in (("realmlp", realmlp), ("tabm_mini", tabm)):
        if not all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ):
            raise AssertionError(f"{name} produced a non-finite gradient")

    payload = {
        "status": "passed",
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "realmlp_output_shape": list(realmlp_prediction.shape),
        "tabm_member_output_shape": list(member_prediction.shape),
        "realmlp_parameters": sum(p.numel() for p in realmlp.parameters()),
        "tabm_mini_parameters": sum(p.numel() for p in tabm.parameters()),
        "tabm_individual_member_loss": float(individual_member_loss.detach().cpu()),
        "tabm_loss_of_mean_prediction": float(loss_of_mean_prediction.detach().cpu()),
        "tabm_member_initial_std": float(
            member_prediction.detach().float().std(dim=1).mean().cpu()
        ),
    }
    output = WORKSPACE_ROOT / "artifacts" / "v2" / "neural_architecture_smoke.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
