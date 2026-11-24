"""Selected VICT R0 adaptation; original numerical AST retained."""
from types import SimpleNamespace
import time,math
import numpy as np
from vict_native import *

def adapt_predict(model,parameters,trainable_names,cpu_base_encoder,optimizer,sentinel,mask,device,tiles,ep):
    import torch
    import torch_xla.core.xla_model as xm
    from torch_xla.amp import GradScaler,autocast
    args=SimpleNamespace(reset_storage='cpu')
    base_encoder=None
    diagnostic_steps=0
    with torch.no_grad():
        if args.reset_storage == 'device':
            for name in trainable_names:
                parameters[name].copy_(base_encoder[name])
        else:
            transferred = xm.send_cpu_data_to_device([cpu_base_encoder[name] for name in trainable_names], device)
            for (name, source) in zip(trainable_names, transferred):
                parameters[name].copy_(source)
        reset_optimizer_state(optimizer)
    xm.mark_step()
    xm.wait_device_ops()
    if args.reset_storage == 'cpu':
        del transferred
        xm.mark_step()
        xm.wait_device_ops()
    sentinel_before = sentinel.detach().flatten()[:1024].cpu()
    (losses, grad_norms, step_times) = ([], [], [])
    found_inf_values = []
    scaler = GradScaler(init_scale=AMP_INITIAL_SCALE)
    model.train(True)
    for step in range(STEPS):
        (samples_np, targets_np) = make_canvas(tiles, ep['task'], None)
        (samples, targets) = to_device(samples_np, targets_np, device)
        optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        with autocast(device, dtype=torch.bfloat16):
            support_label = targets[:, :, :448, :]
            latent_query = model.forward_encoder(samples, targets, 0, mask)
            pred_query = model.forward_decoder(latent_query)
            cycle_targets = torch.cat([support_label, pred_query[:, :, 448:, :]], dim=2)
            latent_support = model.forward_encoder(samples, cycle_targets, 2, mask)
            pred_support = model.forward_decoder(latent_support)
            loss = model.forward_loss(pred_support, samples, cycle_targets, 2, mask, torch.ones_like(targets))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = sentinel.grad.detach().float().norm()
        scaler_state = scaler._per_optimizer_states[id(optimizer)]
        found_inf = torch.stack(tuple(scaler_state['found_inf_per_device'].values())).sum()
        scaler.step(optimizer)
        scaler.update()
        xm.mark_step()
        xm.wait_device_ops()
        loss_value = float(loss.detach().cpu())
        grad_value = float(grad_norm.cpu())
        found_inf_value = float(found_inf.detach().cpu())
        found_inf_values.append(found_inf_value)
        if step < diagnostic_steps:
            opt_state = optimizer.state[sentinel]
            diag_tensors = {'sentinel_finite': sentinel.detach().float().isfinite().all(), 'sentinel_absmax': sentinel.detach().float().abs().max(), 'exp_avg_finite': opt_state['exp_avg'].detach().float().isfinite().all(), 'exp_avg_absmax': opt_state['exp_avg'].detach().float().abs().max(), 'exp_avg_sq_finite': opt_state['exp_avg_sq'].detach().float().isfinite().all(), 'exp_avg_sq_absmax': opt_state['exp_avg_sq'].detach().float().abs().max(), 'optimizer_step': opt_state['step'].detach().float()}
            diag = {name: float(value.cpu()) for (name, value) in diag_tensors.items()}
            print('VICT_NUMERIC_DIAGNOSTIC ' + json.dumps({'step': step + 1, 'loss': loss_value, 'sentinel_grad_norm': grad_value, 'found_inf': found_inf_value, **diag}, sort_keys=True), flush=True)
        if not math.isfinite(loss_value) or (not math.isfinite(grad_value) and found_inf_value == 0.0):
            raise SystemExit(f"nonfinite optimization evidence {ep['episode_id']} step={step + 1} loss={loss_value!r} sentinel_grad_norm={grad_value!r} found_inf={found_inf_value!r}")
        losses.append(loss_value)
        if math.isfinite(grad_value):
            grad_norms.append(grad_value)
        step_times.append(time.perf_counter() - started)
        if step in (0, STEPS - 1) or (step + 1) % 10 == 0:
            print(f"VICT_STEP episode={ep['episode_id']} step={step + 1}/{STEPS} loss={loss_value:.8f} grad={grad_value:.8f} found_inf={found_inf_value:.0f} seconds={step_times[-1]:.3f}", flush=True)
    sentinel_after = sentinel.detach().flatten()[:1024].cpu()
    parameter_delta = float((sentinel_after - sentinel_before).abs().max())
    if max(grad_norms) <= 0 or not math.isfinite(parameter_delta) or parameter_delta <= 0:
        raise SystemExit(f"no proven encoder update for {ep['episode_id']}")
    (infer_np, infer_target_np) = make_canvas(tiles, ep['task'], None)
    (infer_x, infer_y) = to_device(infer_np, infer_target_np, device)
    inference_started = time.perf_counter()
    prediction = predict(model, infer_x, infer_y, mask, ep['task'])
    inference_seconds = time.perf_counter() - inference_started
    return prediction,dict(steps=STEPS,losses=losses,finite_sentinel_grad_norms=grad_norms,found_inf_values=found_inf_values,parameter_delta=parameter_delta,scaler_final_scale=float(scaler.get_scale()),step_seconds=step_times,inference_seconds=inference_seconds)
