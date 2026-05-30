#!/usr/bin/env python3
"""Bridge existing Ropedia sensor-adapter tokens to Qwen3-Omni hidden size."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def parse_args() -> argparse.Namespace:
    workspace_default = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Create Qwen-sized soft tokens from a trained sensor adapter.")
    parser.add_argument("--run-id", default="qwen_lora_sensor_bridge")
    parser.add_argument("--sensor-adapter-model", type=Path, default=workspace_default / "results/omni_finetune/adapter_only/adapter_only/sensor_adapter_model.pt")
    parser.add_argument("--qwen-config", type=Path)
    parser.add_argument("--qwen-hidden-size", type=int, default=0, help="0 means infer from --qwen-config, else use 2048.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--feature-npz", type=Path, help="Optional exported feature shard to project for inspection.")
    parser.add_argument("--sample-limit", type=int, default=16)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    return parser.parse_args()


class SensorAdapterEncoder(nn.Module):
    """Architecture-compatible encoder for `qwen3_omni_adapter_smoke.py` artifacts."""

    def __init__(self, block_specs: list[tuple[str, int, int]], hidden_dim: int, n_layers: int):
        super().__init__()
        self.block_specs = block_specs
        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(end - start),
                nn.Linear(end - start, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _name, start, end in block_specs
        ])
        self.type_embedding = nn.Parameter(torch.randn(len(block_specs), hidden_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=max(1, min(8, hidden_dim // 32)),
            dim_feedforward=hidden_dim * 4,
            dropout=0.10,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        tokens = []
        for adapter, (_name, start, end) in zip(self.adapters, self.block_specs):
            tokens.append(adapter(features[:, start:end]))
        x = torch.stack(tokens, dim=1) + self.type_embedding.unsqueeze(0)
        return self.fusion(x)


class SensorToQwenBridge(nn.Module):
    def __init__(self, encoder: SensorAdapterEncoder, adapter_hidden_dim: int, qwen_hidden_size: int):
        super().__init__()
        self.encoder = encoder
        self.proj = nn.Sequential(
            nn.LayerNorm(adapter_hidden_dim),
            nn.Linear(adapter_hidden_dim, qwen_hidden_size),
            nn.LayerNorm(qwen_hidden_size),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.proj(self.encoder(features))


def infer_layers(state_dict: dict) -> int:
    layers = set()
    for key in state_dict:
        if key.startswith("fusion.layers."):
            parts = key.split(".")
            if len(parts) > 2 and parts[2].isdigit():
                layers.add(int(parts[2]))
    return max(layers) + 1 if layers else 1


def infer_qwen_hidden_size(args: argparse.Namespace) -> int:
    if args.qwen_hidden_size > 0:
        return args.qwen_hidden_size
    if args.qwen_config and args.qwen_config.exists():
        payload = json.loads(args.qwen_config.read_text(encoding="utf-8"))
        thinker = payload.get("thinker_config", {})
        text_config = thinker.get("text_config", {})
        hidden = text_config.get("hidden_size")
        if hidden:
            return int(hidden)
    return 2048


def load_bridge(args: argparse.Namespace) -> tuple[SensorToQwenBridge, dict]:
    artifact = torch.load(args.sensor_adapter_model, map_location="cpu")
    blocks = [(str(name), int(start), int(end)) for name, start, end in artifact["blocks"]]
    state_dict = artifact["model_state"]
    hidden_dim = int(state_dict["type_embedding"].shape[1])
    n_layers = infer_layers(state_dict)
    encoder = SensorAdapterEncoder(blocks, hidden_dim, n_layers)
    missing, unexpected = encoder.load_state_dict(state_dict, strict=False)
    qwen_hidden = infer_qwen_hidden_size(args)
    bridge = SensorToQwenBridge(encoder, hidden_dim, qwen_hidden)
    metadata = {
        "sensor_adapter_model": str(args.sensor_adapter_model),
        "qwen_hidden_size": qwen_hidden,
        "adapter_hidden_dim": hidden_dim,
        "num_adapter_tokens": len(blocks),
        "blocks": [{"name": name, "start": start, "end": end, "dim": end - start} for name, start, end in blocks],
        "load_missing_keys": missing,
        "load_unexpected_keys": unexpected,
        "note": "Projection is intentionally separate from native LoRA. Use it after Qwen video/audio/text SFT is validated.",
    }
    return bridge, metadata


def main() -> int:
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = Path(__file__).resolve().parents[2] / "checkpoints" / args.run_id / "sensor_bridge"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    bridge, metadata = load_bridge(args)
    bridge.to(device).eval()
    torch.save({"state_dict": bridge.state_dict(), "metadata": metadata}, args.output_dir / "sensor_to_qwen_bridge.pt")
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if args.feature_npz:
        data = np.load(args.feature_npz, allow_pickle=True)
        features = data["features"].astype(np.float32)
        if args.sample_limit > 0:
            features = features[: args.sample_limit]
        with torch.no_grad():
            tokens = bridge(torch.from_numpy(features).to(device)).detach().cpu().numpy()
        np.savez_compressed(args.output_dir / "projected_sensor_tokens.npz", tokens=tokens)
        metadata["projected_tokens_shape"] = list(tokens.shape)
        (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    report = [
        "# Qwen3-Omni Sensor Bridge",
        "",
        f"- Sensor adapter: `{args.sensor_adapter_model}`",
        f"- Qwen hidden size: `{metadata['qwen_hidden_size']}`",
        f"- Adapter tokens: `{metadata['num_adapter_tokens']}`",
        "",
        "This bridge converts the existing Ropedia adapter tokens into Qwen-sized soft tokens. It is opt-in scaffolding for the post-native-LoRA phase, where these tokens can be consumed through prefix insertion or a cross-attention memory bridge.",
    ]
    (args.output_dir / "RUN_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
