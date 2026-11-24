"""Sensitivity variant: remove only the negative-scale reset in depth_midas.

This is a different evaluation protocol, not an inference-time decoder. It fits
scale and shift (including the sign) using each query's ground truth. Preserve
the original invalid mask, minimum inverse depth, coverage and depth metrics.
"""
import numpy as np

CONTRACT='bench_v2_depth_signed_scale_sensitivity_v1'


def depth_signed_scale(pred_grey,pred_invalid,gt_inv,gt_valid,*,min_inv):
    gv=np.asarray(gt_valid,bool);pv=gv & ~np.asarray(pred_invalid,bool)
    out={'n_valid':int(gv.sum()),'coverage':float(pv.sum()/max(gv.sum(),1))}
    if gv.sum()==0:
        return out|{k:float('nan') for k in ('delta1','delta2','delta3','abs_rel','rmse','scale','shift')}
    x=np.asarray(pred_grey,np.float64)[pv];y=np.asarray(gt_inv,np.float64)[pv]
    if x.size>=2 and np.ptp(x)>0:
        A=np.stack([x,np.ones_like(x)],1)
        s,t=np.linalg.lstsq(A,y,rcond=None)[0]
    else:
        s,t=0.,float(y.mean()) if y.size else float(min_inv)
    # Intentionally retain a negative s. Everything after fitting is unchanged.
    aligned=np.full(gv.sum(),float(min_inv))
    aligned[pv[gv]]=s*np.asarray(pred_grey,np.float64)[pv]+t
    aligned=np.maximum(aligned,min_inv)
    dp=1./aligned;dg=1./np.asarray(gt_inv,np.float64)[gv]
    ratio=np.maximum(dp/dg,dg/dp)
    return out|{'delta1':float(np.mean(ratio<1.25)),
        'delta2':float(np.mean(ratio<1.25**2)),'delta3':float(np.mean(ratio<1.25**3)),
        'abs_rel':float(np.mean(np.abs(dp-dg)/dg)),'rmse':float(np.sqrt(np.mean((dp-dg)**2))),
        'scale':float(s),'shift':float(t)}
