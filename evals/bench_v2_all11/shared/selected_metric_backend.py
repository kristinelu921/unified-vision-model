"""Explicit saved-recipe evaluation amendments around the unchanged base scorer."""
import copy
import hashlib
import importlib.util
import math
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parent
SITE=ROOT.parents[2]/'visualcloze-results-site'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name,path,expected):
    assert sha(path)==expected, 'Saved metric amendment changed'
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


class SelectedBackend:
    metric_contract='eval_v2'

    def __init__(self,base,model):
        self.base,self.model=base,model
        self.paths={SITE/'dist/bench-v2-signed-depth/signed_depth_metric.py':'0dda801d349f87ec2a09b09b818c98917372e3687a6849cb3250d7e3ff92c4b9',
                    SITE/'scripts/depth_output_adapter.py':'b7283b8e45263fb6952fcbb0673d3578173f88db155c5329ff43f02d38398b8d',
                    SITE/'scripts/edge_contrast_adapter.py':'ca93b7ed369ada346c373949cb971afcbf67758a3c3af7346210ce48c77bb26c'}
        self.signed=load('selected_signed_depth',list(self.paths)[0],list(self.paths.values())[0])
        self.depth=load('selected_native_depth',list(self.paths)[1],list(self.paths.values())[1])
        self.edge=load('selected_improv_edge',list(self.paths)[2],list(self.paths.values())[2])
        self.paths[Path(__file__)]=sha(Path(__file__))
        self.identity=copy.deepcopy(base.identity)
        self.identity.update(backend_contract='bench_all11_selected_recipe_backend_v1',
            evaluation_contract=self.signed.CONTRACT,selected_model=model,
            selected_amendment_source_sha256={str(p):v for p,v in self.paths.items()},
            native_depth_decoder=self.depth.CONTRACT if model in self.depth.MODELS else None,
            edge_postprocessing=self.edge.CONTRACT if model=='improv' else None,
            metric_protocol_status='frozen base with explicitly versioned selected-recipe amendments')

    def validate_runtime_identity(self):
        self.base.validate_runtime_identity()
        assert all(sha(p)==expected for p,expected in self.paths.items())

    def dataset_edge_readout(self,*args,**kwargs):
        return self.base.dataset_edge_readout(*args,**kwargs)

    def score_pair(self,task,pred,target,source,*,context):
        import numpy as np
        copy_control=context['prediction_stem']=='query_input'
        if task!='relative_depth' and not (task=='edge_detection' and self.model=='improv' and not copy_control):
            return self.base.score_pair(task,pred,target,source,context=context)
        self.validate_runtime_identity()
        ep=context['episode']
        assert ep['task']==task and ep['query_source']==source
        assert pred.dtype==target.dtype==np.uint8 and pred.shape==target.shape==(256,256,3)
        ctx=dict(self.base._episode_context(ep,target))
        if copy_control:assert np.array_equal(pred,ctx['input'])
        P=self.base.corrected.protocol
        if task=='relative_depth':
            scored=ctx['gt_scored'] & P.depth_eval_region(ctx['dataset'],ctx['target'].shape)
            limits=P.DEPTH_RANGE_M.get(ctx['dataset'])
            if limits:
                with np.errstate(divide='ignore'):depth=1./np.asarray(ctx['gt_inv'],np.float64)
                scored=scored & (depth>=limits[0]) & (depth<=limits[1])
            native_decode=self.model in self.depth.MODELS and not copy_control
            grey,invalid=(self.depth.decode_depth_prediction(pred,model=self.model) if native_decode
                          else self.base.corrected.pp.depth_grey(pred))
            values=self.signed.depth_signed_scale(grey,invalid,ctx['gt_inv'],scored,min_inv=P.depth_min_inv(ctx['dataset'],ctx))
            values['n_scored']=values.pop('n_valid')
            values.update(depth_evaluation_contract=self.signed.CONTRACT,
                          native_depth_decode_applied=native_decode)
        else:
            confidence=self.edge.improv_edge_confidence(pred)
            dataset=ctx['dataset'];tol=P.edge_tolerance_px(dataset,ctx['target'].shape)
            if dataset in ('bsds500','bsds500_test'):
                rel,pack=self.base._get_pack(ep)
                annotators=self.base._bsds_annotators(rel,pack,ep['query_record'])
                counts=self.base.corrected.adapter.edge_counts_multi(dataset,confidence,annotators,tol)
            elif dataset=='nyudv2_test_hed':
                counts=self.base.corrected.hk.edge_counts_multi(confidence,[ctx['gt_edge_thin']],tol)
            else:raise ValueError('Unregistered edge dataset')
            result=self.base.corrected.metrics.ois(counts)
            values=dict(ois_f=result['f'],ois_precision=result['precision'],ois_recall=result['recall'],
                        _edge_counts=counts.tolist(),tol_px=tol,edge_postprocessing_contract=self.edge.CONTRACT)
        name,lower=P.PRIMARY[task];primary=float(values[name]);assert math.isfinite(primary)
        self.validate_runtime_identity()
        return dict(values,native=primary,native_name=name,native_lower=bool(lower),psnr=float('nan'),
                    mae=float(np.abs(pred.astype(np.float64)-target.astype(np.float64)).mean()/255))


def create_backend():
    from tools.bench_eval_v3_backend_v1 import create_backend as frozen
    model=os.environ['BENCH_SELECTED_MODEL']
    if model not in {'visualcloze','painter','vict','genlv','lavin_dit','prompt_diffusion','relation_adapter','instamanip','improv','analogist','lvm'}:
        raise ValueError('Unknown selected model')
    return SelectedBackend(frozen(),model)
