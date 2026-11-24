#!/usr/bin/env python3
"""Apply the audited dynamic-scalar XLA patch to the pinned pipeline."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path


UPSTREAM_SHA256 = "ffe4c323e55746c4a55e6afe3aa0bceec1de66a681e8780765d6294ee8deda3e"
PATCHED_SHA256 = "b7efc22c4feaae1ef153f6664b49c9141d322b26701a2a2af89734ff333a425c"

REPLACEMENTS = (
    (
        """            for i, t in enumerate(timesteps):
                # Expand the latents if we are doing classifier free guidance.
""",
        """            for i, t in enumerate(timesteps):
                if getattr(self.scheduler, "_instamanip_dynamic_euler_schema", None) == "instamanip_xla_dynamic_euler_device_scalars_v1":
                    t = self.scheduler.instamanip_timestep_to_device(t, latents)
                # Expand the latents if we are doing classifier free guidance.
""",
    ),
    (
        """                if scheduler_is_in_sigma_space:
                    step_index = (self.scheduler.timesteps == t).nonzero()[0].item()
                    sigma = self.scheduler.sigmas[step_index]
                    noise_pred = latent_model_input - sigma * noise_pred
""",
        """                if scheduler_is_in_sigma_space:
                    if getattr(self.scheduler, "_instamanip_dynamic_euler_schema", None) == "instamanip_xla_dynamic_euler_device_scalars_v1":
                        sigma = self.scheduler.instamanip_current_sigma()
                    else:
                        step_index = (self.scheduler.timesteps == t).nonzero()[0].item()
                        sigma = self.scheduler.sigmas[step_index]
                    noise_pred = latent_model_input - sigma * noise_pred
""",
    ),
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def transform(source: str) -> str:
    value = source
    for old, new in REPLACEMENTS:
        if value.count(old) != 1:
            raise ValueError("audited dynamic diffusion patch context differs")
        value = value.replace(old, new)
    return value


def apply(path: Path) -> str:
    path = Path(path)
    before = sha256(path)
    if before == PATCHED_SHA256:
        return before
    if before != UPSTREAM_SHA256:
        raise ValueError(f"refusing unpinned upstream pipeline: {before}")
    value = transform(path.read_text())
    if sha256_bytes(value.encode()) != PATCHED_SHA256:
        raise ValueError("patched pipeline does not match the audited identity")
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(value)
    os.replace(temporary, path)
    return sha256(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pipeline", type=Path)
    args = parser.parse_args()
    print(apply(args.pipeline))


if __name__ == "__main__":
    main()
