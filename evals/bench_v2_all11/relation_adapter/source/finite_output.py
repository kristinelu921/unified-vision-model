"""Inspect raw decoded floats before postprocessing can hide infinities."""
import hashlib
import torch


def finite_cpu(value, *, stage, report):
    array = value.detach().float().cpu()
    finite = torch.isfinite(array)
    if not bool(finite.all().item()):
        report({"status": "NONFINITE_FLOAT", "stage": stage,
                "shape": list(value.shape), "dtype": str(value.dtype),
                "nan_count": int(torch.isnan(array).sum().item()),
                "positive_infinity_count": int(torch.isposinf(array).sum().item()),
                "negative_infinity_count": int(torch.isneginf(array).sum().item())})
        raise ValueError("Nonfinite " + stage + "; no valid prediction may be emitted")
    return array


def install_decode_guard(vae, report):
    original = vae.decode
    state = {"enabled": True, "checked_decodes": 0, "stage": "raw_vae_decode_before_postprocess"}

    def guarded(*args, **kwargs):
        result = original(*args, **kwargs)
        tensor = result[0] if isinstance(result, (tuple, list)) else result.sample
        finite_cpu(tensor, stage=state["stage"], report=report)
        state["checked_decodes"] += 1
        return result

    vae.decode = guarded
    return state


def float_digest(array):
    return hashlib.sha256(array.contiguous().numpy().tobytes()).hexdigest()
