#!/usr/bin/env python3
"""Export the MobileI2V image encoder and Turbo-VAED video decoder.

The spatial axes remain dynamic so one encoder and one decoder serve both
native 16:9 and 9:16 layouts.  The denoiser itself is exported separately per
layout because its rotary position grid is layout-specific.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_onnx_transpose_axes(path: Path) -> int:
    """Replace negative Transpose permutation entries for ORT portability."""
    import onnx

    model = onnx.load(path)
    changed = 0
    for node in model.graph.node:
        if node.op_type != "Transpose":
            continue
        for attribute in node.attribute:
            if attribute.name != "perm":
                continue
            rank = len(attribute.ints)
            normalized = [value + rank if value < 0 else value for value in attribute.ints]
            if normalized != list(attribute.ints):
                del attribute.ints[:]
                attribute.ints.extend(normalized)
                changed += 1
    if changed:
        onnx.save(model, path)
    return changed


class LtxImageEncoder(torch.nn.Module):
    def __init__(self, vae: torch.nn.Module) -> None:
        super().__init__()
        self.vae = vae

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        posterior = self.vae.encode(image).latent_dist
        # MobileI2V was trained with posterior sampling. Android performs the
        # sampling itself so a job seed can be checkpointed and reproduced.
        return posterior.mean, posterior.logvar


class TurboVideoDecoder(torch.nn.Module):
    def __init__(self, decoder: torch.nn.Module) -> None:
        super().__init__()
        self.decoder = decoder

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)


def export_encoder(model_dir: Path, output: Path) -> None:
    from diffusers import AutoencoderKLLTXVideo

    vae = AutoencoderKLLTXVideo.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        local_files_only=True,
    ).eval()
    wrapper = LtxImageEncoder(vae).eval()
    # Spatial axes are dynamic. A compact exemplar avoids spending minutes on
    # a redundant 720p eager forward during tracing; production dimensions are
    # validated independently with ONNX Runtime after export.
    image = torch.zeros(1, 3, 1, 64, 96, dtype=torch.float16)
    print(f"parameters={sum(p.numel() for p in vae.encoder.parameters())}", flush=True)
    torch.onnx.export(
        wrapper,
        (image,),
        output,
        input_names=["image"],
        output_names=["latent_mean", "latent_logvar"],
        dynamic_axes={
            "image": {3: "image_height", 4: "image_width"},
            "latent_mean": {3: "latent_height", 4: "latent_width"},
            "latent_logvar": {3: "latent_height", 4: "latent_width"},
        },
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
        external_data=True,
    )
    print(f"normalized_transposes={normalize_onnx_transpose_axes(output)}", flush=True)


def export_decoder(source_root: Path, config_path: Path, checkpoint: Path, output: Path) -> None:
    sys.path.insert(0, str(source_root))
    from diffusers_vae.src.diffusers.models.autoencoders.autoencoder_kl_turbo_vaed import (
        AutoencoderKLTurboVAED,
    )

    with config_path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    vae = AutoencoderKLTurboVAED.from_config(config).eval().half()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    incompatible = vae.decoder.load_state_dict(state, strict=False)
    if incompatible.missing_keys:
        raise RuntimeError(f"Decoder checkpoint is incomplete: {incompatible.missing_keys}")
    wrapper = TurboVideoDecoder(vae.decoder).eval()
    latent = torch.zeros(1, 128, 3, 2, 3, dtype=torch.float16)
    print(f"checkpoint_sha256={sha256(checkpoint)}", flush=True)
    print(f"parameters={sum(p.numel() for p in vae.decoder.parameters())}", flush=True)
    torch.onnx.export(
        wrapper,
        (latent,),
        output,
        input_names=["latent"],
        output_names=["video"],
        dynamic_axes={
            "latent": {3: "latent_height", 4: "latent_width"},
            "video": {3: "video_height", 4: "video_width"},
        },
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
        external_data=True,
    )
    print(f"normalized_transposes={normalize_onnx_transpose_axes(output)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--component", choices=("encoder", "decoder"), required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.component == "encoder":
        if args.model_dir is None:
            parser.error("--model-dir is required for the encoder")
        export_encoder(args.model_dir, args.output)
    else:
        required = (args.source_root, args.config, args.checkpoint)
        if any(value is None for value in required):
            parser.error("--source-root, --config and --checkpoint are required for the decoder")
        export_decoder(args.source_root, args.config, args.checkpoint, args.output)

    print(f"onnx={args.output} bytes={args.output.stat().st_size}", flush=True)
    print(f"onnx_sha256={sha256(args.output)}", flush=True)


if __name__ == "__main__":
    main()
