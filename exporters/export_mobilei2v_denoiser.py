#!/usr/bin/env python3
"""Export the released MobileI2V denoiser to a static-shape ONNX graph.

This exporter intentionally produces one-sample graphs. Android performs
classifier-free guidance as two sequential invocations, which lowers peak RAM
on an 8 GB phone at the cost of extra render time.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class AndroidDenoiser(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, latent_height: int, latent_width: int) -> None:
        super().__init__()
        self.model = model
        token_count = 3 * latent_height * latent_width
        first_frame_tokens = latent_height * latent_width
        condition_mask = torch.zeros(1, token_count, dtype=torch.float16)
        condition_mask[:, :first_frame_tokens] = 1
        self.register_buffer("condition_mask", condition_mask, persistent=False)

    def forward(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        caption: torch.Tensor,
        flow_score: torch.Tensor,
    ) -> torch.Tensor:
        # guide_image is unused inside the released denoiser; the scheduler
        # re-injects the encoded first frame after every Euler step.
        return self.model(
            latent,
            timestep,
            latent[:, :, :1],
            caption,
            self.condition_mask,
            flow_score,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--orientation", choices=("landscape", "portrait"), default="landscape")
    args = parser.parse_args()

    sys.path.insert(0, str(args.source_root))
    from diffusion.model.builder import build_model
    import diffusion.model.nets  # noqa: F401 - registers Mobiledit

    latent_height, latent_width = ((23, 40) if args.orientation == "landscape" else (40, 23))
    model = build_model(
        "Mobiledit_300M_P1_D16",
        model_max_length=300,
        qk_norm=True,
        caption_channels=896,
        y_norm=True,
        attn_type="linear",
        ffn_type="glumbconv",
        mlp_ratio=2.5,
        mlp_acts=["silu", "silu", None],
        in_channels=128,
        y_norm_scale_factor=0.01,
        use_pe=False,
        linear_head_dim=32,
        pred_sigma=False,
        learn_sigma=False,
        cross_norm=False,
        use_fp32_attention=True,
    )
    for module in model.modules():
        if hasattr(module, "position_getter") and hasattr(module, "rope"):
            module.mobile_video_layout = (3, latent_height, latent_width)

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    state = checkpoint["state_dict"]
    state.pop("pos_embed", None)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing != ["pos_embed"] or unexpected:
        raise RuntimeError(f"Checkpoint mismatch; missing={missing}, unexpected={unexpected}")

    wrapper = AndroidDenoiser(model.eval().half(), latent_height, latent_width).eval()
    latent = torch.zeros(1, 128, 3, latent_height, latent_width, dtype=torch.float16)
    timestep = torch.tensor([500.0], dtype=torch.float16)
    caption = torch.zeros(1, 1, 300, 896, dtype=torch.float16)
    flow_score = torch.tensor([1.0], dtype=torch.float16)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"checkpoint_sha256={sha256(args.checkpoint)}", flush=True)
    print(f"parameters={sum(p.numel() for p in model.parameters())}", flush=True)
    torch.onnx.export(
        wrapper,
        (latent, timestep, caption, flow_score),
        args.output,
        input_names=["latent", "timestep", "caption", "flow_score"],
        output_names=["velocity"],
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
        external_data=True,
    )
    print(f"onnx={args.output} bytes={args.output.stat().st_size}", flush=True)


if __name__ == "__main__":
    main()
