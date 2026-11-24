"""Persistent inference components with per-batch immutable conditioning reuse.

No pack, plan, scorer, targets, retrieval or candidate selection is imported.
The only learned conditions are the permitted pair/query and fixed text.
"""
from __future__ import annotations
import math
import time
import torch

def schedule_rows(scheduler, steps=50):
    scheduler.set_timesteps(steps, device='cpu')
    if scheduler.config.prediction_type != 'epsilon' or scheduler.config.clip_sample:
        raise ValueError('handwritten baseline only implements unclipped epsilon DDIM')
    if getattr(scheduler.config, 'thresholding', False):
        raise ValueError('thresholding is outside the frozen DDIM recipe')
    if float(scheduler.init_noise_sigma) != 1.:
        raise ValueError('frozen initialization assumes unit noise sigma')
    alphas=scheduler.alphas_cumprod.cpu()
    stride=scheduler.config.num_train_timesteps//steps
    rows=[]
    for t in scheduler.timesteps.tolist():
        a=float(alphas[t]); previous=t-stride
        ap=float(alphas[previous]) if previous>=0 else float(scheduler.final_alpha_cumprod)
        rows.append((int(t),math.sqrt(1.-a),math.sqrt(a),math.sqrt(ap),math.sqrt(1.-ap)))
    return rows

def ddim_step(sample, predicted, coefficients, preserve_scalar_opmath=False):
    beta_sqrt, alpha_sqrt, previous_alpha_sqrt, previous_beta_sqrt=coefficients
    if preserve_scalar_opmath and sample.dtype in (torch.bfloat16,torch.float16):
        # CPU scalar operations and TPU scalar multiplications use FP32 opmath.
        # The frozen TPU BF16 scalar division lowers differently; retain that
        # dtype and all operation boundaries, as verified on fixed fixtures.
        product=(predicted.float()*beta_sqrt).to(sample.dtype)
        numerator=sample-product
        if sample.device.type == 'xla' and sample.dtype == torch.bfloat16 and torch.is_tensor(alpha_sqrt) and alpha_sqrt.dtype == sample.dtype:
            # Matches the frozen TPU scalar division's BF16 lowering.
            original=numerator/alpha_sqrt
        else:
            original=(numerator.float()/alpha_sqrt).to(sample.dtype)
        left=(original.float()*previous_alpha_sqrt).to(sample.dtype)
        right=(predicted.float()*previous_beta_sqrt).to(sample.dtype)
        return left+right
    original=(sample-beta_sqrt*predicted)/alpha_sqrt
    return previous_alpha_sqrt*original+previous_beta_sqrt*predicted

class Engine:
    def __init__(self, controlnet, unet, vae, scheduler, *, device, dtype,
                 cache_conditioning=False, dynamic_coefficients=False,
                 barrier=lambda:None, shard=lambda x:None):
        self.controlnet,self.unet,self.vae=controlnet.eval(),unet.eval(),vae.eval()
        self.scheduler,self.device,self.dtype=scheduler,device,dtype
        self.cache_conditioning=cache_conditioning
        self.dynamic_coefficients=dynamic_coefficients
        self.barrier,self.shard=barrier,shard
        self.rows=schedule_rows(scheduler)
        self.tpu_scalar_rounding=(dynamic_coefficients and torch.device(device).type == 'xla' and dtype == torch.bfloat16)
        self.device_rows=None
        if dynamic_coefficients:
            # Materialize buffers before the loop so timestep/coefficients enter
            # compiled steps as device inputs instead of 50 Python constants.
            self.device_rows=[(torch.tensor(t,dtype=torch.int64).to(device),
                tuple(torch.tensor(c,dtype=dtype if self.tpu_scalar_rounding and j==1 else torch.float32).to(device) for j,c in enumerate(cs)))
                for t,*cs in self.rows]
            barrier()

    @torch.no_grad()
    def denoise(self, latents, pair, query, positive, negative, guidance=7.5):
        batch=latents.shape[0]
        if positive.shape[0]!=batch or pair.shape[:2]!=(batch,6) or query.shape[:2]!=(batch,3):
            raise ValueError('conditioning batch or channel order differs')
        if negative.shape[0]==1:
            negative=negative.repeat(batch,1,1)
        embeddings=torch.cat([negative,positive])
        pair=torch.cat([pair,pair]);query=torch.cat([query,query])
        for tensor in (latents,pair,query,embeddings):self.shard(tensor)
        if self.cache_conditioning:
            if self.controlnet.config.controlnet_conditioning_channel_order!='rgb':
                raise ValueError('cached conditioning supports the frozen RGB model only')
            pair=self.controlnet.controlnet_cond_embedding(pair)
            query=self.controlnet.controlnet_query_cond_embedding(query)
            self.shard(pair);self.shard(query)
            self.barrier()
        for i,(t,*coefficients) in enumerate(self.rows):
            terminal_scalar=(self.tpu_scalar_rounding and i == len(self.rows)-1)
            if self.dynamic_coefficients:
                timestep,device_coefficients=self.device_rows[i]
                # Preserve original scalar execution at the public DDIM terminal boundary.
                if not terminal_scalar:coefficients=device_coefficients
            else:
                timestep=torch.tensor(t,device=self.device,dtype=torch.int64)
            latent_input=torch.cat([latents,latents]);self.shard(latent_input)
            kwargs={}
            if self.cache_conditioning:kwargs['conditioning_precomputed']=True
            down,mid=self.controlnet(latent_input,timestep,encoder_hidden_states=embeddings,
                controlnet_cond=pair,controlnet_query_cond=query,conditioning_scale=1.,
                guess_mode=False,return_dict=False,**kwargs)
            guided=self.unet(latent_input,timestep,encoder_hidden_states=embeddings,
                down_block_additional_residuals=down,mid_block_additional_residual=mid,
                return_dict=False)[0]
            unconditioned,conditioned=guided.chunk(2)
            predicted=unconditioned+guidance*(conditioned-unconditioned)
            latents=ddim_step(latents,predicted,coefficients,
                             preserve_scalar_opmath=self.dynamic_coefficients and not terminal_scalar)
            self.shard(latents);self.barrier()
        return latents

    @torch.no_grad()
    def predict(self, demo_input, demo_output, query, noise, positive, negative):
        def transfer(value):
            return torch.as_tensor(value).to(device=self.device,dtype=self.dtype)
        di,do,q,latents,p,n=map(transfer,(demo_input,demo_output,query,noise,positive,negative))
        latent=self.denoise(latents,torch.cat([di,do],dim=1),q,p,n)
        decoded=self.vae.decode(latent/float(self.vae.config.scaling_factor),return_dict=False)[0]
        decoded=(decoded/2.+.5).clamp(0.,1.)
        self.barrier()
        pixels=(decoded.float().cpu()*255.).round().clamp(0,255).to(torch.uint8)
        return pixels.permute(0,2,3,1).numpy()
