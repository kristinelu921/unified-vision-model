#!/usr/bin/env python3
"""Output-equivalent EulerDiscreteScheduler adapter for XLA graph reuse.

Diffusers 0.25 keeps the Euler schedule on the accelerator and indexes it with
a different Python integer at every denoising step.  PyTorch/XLA consequently
sees a different graph at every step.  This adapter keeps the immutable
schedule on CPU and transfers the current timestep and sigma scalars as device
data, so their values are runtime inputs rather than graph constants.
"""
from __future__ import annotations

import functools
from importlib.metadata import version
import math
import types

import torch


DYNAMIC_EULER_SCHEMA = "instamanip_xla_dynamic_euler_device_scalars_v1"
EXPECTED_DIFFUSERS_VERSION = "0.25.0"
EXPECTED_SCHEDULER_MODULE = "diffusers.schedulers.scheduling_euler_discrete"
EXPECTED_SCHEDULER_CLASS = "EulerDiscreteScheduler"


def euler_scale_model_input(sample: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """The exact Diffusers 0.25 Euler input-scaling expression."""
    return sample / ((sigma**2 + 1) ** 0.5)


def euler_step_values(
    model_output: torch.Tensor,
    sample: torch.Tensor,
    sigma: torch.Tensor,
    sigma_next: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The exact no-churn, epsilon-prediction Diffusers 0.25 Euler equations."""
    sample = sample.to(torch.float32)
    pred_original_sample = sample - sigma * model_output
    derivative = (sample - pred_original_sample) / sigma
    dt = sigma_next - sigma
    prev_sample = sample + derivative * dt
    return prev_sample.to(model_output.dtype), pred_original_sample


def _validate_scheduler(scheduler, expected_steps: int) -> None:
    if version("diffusers") != EXPECTED_DIFFUSERS_VERSION:
        raise RuntimeError("dynamic Euler adapter requires pinned diffusers 0.25.0")
    cls = scheduler.__class__
    if cls.__module__ != EXPECTED_SCHEDULER_MODULE or cls.__name__ != EXPECTED_SCHEDULER_CLASS:
        raise RuntimeError("dynamic Euler adapter requires the exact Diffusers Euler scheduler")
    if int(getattr(scheduler, "order", -1)) != 1:
        raise RuntimeError("dynamic Euler adapter requires scheduler order 1")
    if scheduler.config.prediction_type != "epsilon":
        raise RuntimeError("dynamic Euler adapter requires epsilon prediction")
    if scheduler.config.timestep_type != "discrete":
        raise RuntimeError("dynamic Euler adapter requires discrete timesteps")
    if int(expected_steps) <= 0:
        raise RuntimeError("dynamic Euler adapter requires a positive fixed step count")


def _device_scalar(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if value.device.type != "cpu" or value.ndim != 0 or not value.is_floating_point():
        raise RuntimeError("dynamic Euler schedule values must be floating CPU scalars")
    if reference.device.type not in {"cpu", "xla"}:
        raise RuntimeError("dynamic Euler adapter is restricted to CPU tests and XLA execution")
    # A CPU-to-XLA transfer is represented as DeviceData.  Unlike new_tensor
    # or a Python float, its numeric value is not folded into the compiled HLO.
    return value.detach().clone().to(device=reference.device, dtype=value.dtype)


def install_dynamic_xla_euler(scheduler, *, expected_steps: int):
    """Install the fixed-schedule adapter on one scheduler instance."""
    if getattr(scheduler, "_instamanip_dynamic_euler_schema", None) == DYNAMIC_EULER_SCHEMA:
        if scheduler._instamanip_expected_steps != int(expected_steps):
            raise RuntimeError("dynamic Euler adapter step count changed")
        return scheduler

    _validate_scheduler(scheduler, expected_steps)
    original_set_timesteps = scheduler.set_timesteps

    @functools.wraps(original_set_timesteps)
    def set_timesteps(self, num_inference_steps: int, device=None):
        if int(num_inference_steps) != self._instamanip_expected_steps:
            raise RuntimeError("dynamic Euler adapter received a different diffusion step count")
        # The schedule is tiny and immutable.  Keeping it on CPU prevents a
        # Python integer index from becoming a distinct accelerator constant.
        self._instamanip_original_set_timesteps(num_inference_steps, device="cpu")
        if self.timesteps.device.type != "cpu" or self.sigmas.device.type != "cpu":
            raise RuntimeError("dynamic Euler schedule did not remain on CPU")
        if len(self.timesteps) != num_inference_steps or len(self.sigmas) != num_inference_steps + 1:
            raise RuntimeError("dynamic Euler schedule cardinality differs")
        self._step_index = 0
        self._instamanip_current_sigma = None
        self._instamanip_current_timestep = None

    def timestep_to_device(self, timestep: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        index = self._step_index
        if index is None or not 0 <= int(index) < self._instamanip_expected_steps:
            raise RuntimeError("dynamic Euler timestep index is outside the fixed schedule")
        expected = self.timesteps[int(index)]
        if timestep.device.type != "cpu" or not torch.equal(timestep, expected):
            raise RuntimeError("dynamic Euler timestep order differs from the pinned schedule")
        value = _device_scalar(timestep, reference)
        self._instamanip_current_timestep = value
        return value

    def current_sigma(self) -> torch.Tensor:
        sigma = self._instamanip_current_sigma
        if sigma is None:
            raise RuntimeError("dynamic Euler sigma was requested before scale_model_input")
        return sigma

    def scale_model_input(self, sample: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        index = self._step_index
        if index is None or not 0 <= int(index) < self._instamanip_expected_steps:
            raise RuntimeError("dynamic Euler scale index is outside the fixed schedule")
        if timestep is not self._instamanip_current_timestep:
            raise RuntimeError("dynamic Euler scale received an unaudited timestep tensor")
        sigma = _device_scalar(self.sigmas[int(index)], sample)
        self._instamanip_current_sigma = sigma
        self.is_scale_input_called = True
        return euler_scale_model_input(sample, sigma)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator=None,
        return_dict: bool = True,
    ):
        if (float(s_churn) != 0.0 or float(s_tmin) != 0.0
                or not math.isinf(float(s_tmax)) or float(s_noise) != 1.0):
            raise RuntimeError("dynamic Euler adapter supports only the official no-churn sampler")
        if generator is not None:
            raise RuntimeError("dynamic Euler adapter requires the evaluation's precomputed latents")
        index = self._step_index
        if index is None or not 0 <= int(index) < self._instamanip_expected_steps:
            raise RuntimeError("dynamic Euler step index is outside the fixed schedule")
        if timestep is not self._instamanip_current_timestep:
            raise RuntimeError("dynamic Euler step received an unaudited timestep tensor")
        sigma = self._instamanip_current_sigma
        if sigma is None:
            raise RuntimeError("dynamic Euler step ran before scale_model_input")
        sigma_next = _device_scalar(self.sigmas[int(index) + 1], sample)
        prev_sample, pred_original_sample = euler_step_values(
            model_output, sample, sigma, sigma_next)
        self._step_index = int(index) + 1
        self._instamanip_current_sigma = None
        self._instamanip_current_timestep = None
        if not return_dict:
            return (prev_sample,)
        from diffusers.schedulers.scheduling_euler_discrete import EulerDiscreteSchedulerOutput
        return EulerDiscreteSchedulerOutput(
            prev_sample=prev_sample, pred_original_sample=pred_original_sample)

    scheduler._instamanip_original_set_timesteps = original_set_timesteps
    scheduler._instamanip_expected_steps = int(expected_steps)
    scheduler._instamanip_dynamic_euler_schema = DYNAMIC_EULER_SCHEMA
    scheduler.set_timesteps = types.MethodType(set_timesteps, scheduler)
    scheduler.instamanip_timestep_to_device = types.MethodType(timestep_to_device, scheduler)
    scheduler.instamanip_current_sigma = types.MethodType(current_sigma, scheduler)
    scheduler.scale_model_input = types.MethodType(scale_model_input, scheduler)
    scheduler.step = types.MethodType(step, scheduler)
    return scheduler


def dynamic_euler_audit(scheduler) -> dict:
    if getattr(scheduler, "_instamanip_dynamic_euler_schema", None) != DYNAMIC_EULER_SCHEMA:
        raise RuntimeError("dynamic Euler adapter is not installed")
    return {
        "schema": DYNAMIC_EULER_SCHEMA,
        "diffusers_version": version("diffusers"),
        "scheduler": f"{scheduler.__class__.__module__}.{scheduler.__class__.__name__}",
        "prediction_type": scheduler.config.prediction_type,
        "timestep_type": scheduler.config.timestep_type,
        "steps": scheduler._instamanip_expected_steps,
        "schedule_residency": "cpu",
        "xla_scalar_representation": "device_data",
    }
