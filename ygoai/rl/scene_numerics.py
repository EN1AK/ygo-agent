"""Separate GPU batch-forward drift from same-input scalar reduction errors."""
import math


def log_probs(values):
    peak=max(values);shift=[x-peak for x in values]
    normalizer=math.log(sum(math.exp(x) for x in shift))
    return [x-normalizer for x in shift]


def objective_forward_report(full,micro,parent_full,parent_micro,targets,legal,weights,cc,kc):
    n=len(full);width=len(cc)
    assert width and len(kc)==width and n%width==0
    assert all(len(x)==n for x in [micro,parent_full,parent_micro,targets,legal,weights])
    full_value=micro_value=bound=max_current=max_parent=0.
    for i in range(n):
        w=weights[i];c=cc[i%width];k=kc[i%width]
        assert math.isfinite(w) and w>=0 and math.isfinite(c) and c>=0 and math.isfinite(k) and k>=0
        if not w or not (c or k):continue
        x,y,a,b=full[i],micro[i],parent_full[i],parent_micro[i]
        assert len(x)==len(y)==len(a)==len(b)==len(legal[i]) and all(math.isfinite(z) for row in [x,y,a,b] for z in row)
        support=[j for j,allowed in enumerate(legal[i]) if allowed];assert support and targets[i] in support
        dx=max(abs(u-v) for u,v in zip(x,y));da=max(abs(a[j]-b[j]) for j in support)
        max_current=max(max_current,dx);max_parent=max(max_parent,da)
        # CE uses every model logit, exactly like the actual training function.
        cx,cy=log_probs(x),log_probs(y)
        # KL has the native-legal support explicitly masked in training.
        px,py=log_probs([x[j] for j in support]),log_probs([y[j] for j in support])
        qa,qb=log_probs([a[j] for j in support]),log_probs([b[j] for j in support])
        pf=[math.exp(z) for z in px];pm=[math.exp(z) for z in py]
        klf=sum(p*(l-q) for p,l,q in zip(pf,px,qa));klm=sum(p*(l-q) for p,l,q in zip(pm,py,qb))
        # Expand D(p||q)-D(p'||q') into a log-ratio difference and
        # a probability-mass difference; triangle inequality gives this bound.
        kb=sum(p*abs((l-q)-(v-r))+abs(p-t)*abs(v-r) for p,t,l,q,v,r in zip(pf,pm,px,qa,py,qb))
        full_value+=w*(c*-cx[targets[i]]+k*klf);micro_value+=w*(c*-cy[targets[i]]+k*klm)
        bound+=w*(c*2*dx+k*kb)  # CE is2-Lipschitz in the logit infinity norm.
    assert abs(full_value-micro_value)<=bound+1e-12*max(1.,abs(full_value),abs(micro_value))
    return dict(reference_full=full_value,reference_micro=micro_value,forward_loss_bound=bound,
                max_current_logit_error=max_current,max_parent_logit_error=max_parent)


def qualify_objective(full_value,micro_value,report):
    tolerance=8*2**-23
    full_scale=max(1.,abs(full_value),abs(report['reference_full']))
    micro_scale=max(1.,abs(micro_value),abs(report['reference_micro']))
    ef=abs(full_value-report['reference_full'])/full_scale;em=abs(micro_value-report['reference_micro'])/micro_scale
    assert report['max_current_logit_error']<.002 and report['max_parent_logit_error']<.002,report
    assert ef<=tolerance and em<=tolerance,(ef,em,report)
    allowed=report['forward_loss_bound']+tolerance*(full_scale+micro_scale)
    assert abs(full_value-micro_value)<=allowed,(full_value,micro_value,allowed)
    return dict(passed=True,**report,full_same_input_relative_error=ef,micro_same_input_relative_error=em,
                scalar_difference=abs(full_value-micro_value),derived_difference_bound=allowed,
                same_input_relative_tolerance=tolerance,
                scope='Measured actual qualification forwards and float64 same-input CE/KL references. Training objective and per-update8-epsilon decomposition are unchanged.')
