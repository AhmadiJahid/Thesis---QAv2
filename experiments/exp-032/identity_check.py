"""Identity check for exp-032: every committed own-eval value of each arm vs its exp-032 re-score."""
import json, math
M='/cta/users/fyilmaz/Thesis---QAv2'
CARD=['break_exact_match_rate','sari_macro','ged_macro','chain_validity_macro','hop_count_exact_match_rate','under_decomposition_rate','over_decomposition_rate','step_f1_macro']
def load(p): return json.load(open(p))
def get(d,path):
    for k in path.strip('/').split('/'):
        d=d[k]
    return d
def flat(d):
    """numeric leaves at top level + report_card terms list"""
    out={}
    for k,v in d.items():
        if isinstance(v,(int,float)) and not isinstance(v,bool): out[k]=v
    rc=d.get('decomposition_report_card')
    if isinstance(rc,dict) and isinstance(rc.get('terms'),list):
        for t in rc['terms']:
            if 'metric' in t and isinstance(t.get('value'),(int,float)): out.setdefault(t['metric'],t['value'])
    return out
SRC={
 'exp-008':[('experiments/exp-008/metrics.json','/eval_metrics'),('experiments/exp-016/metrics.json','/arms/exp008_full_train'),('experiments/exp-011/metrics.json','/arms/exp008_full_train'),('experiments/exp-015/metrics.json','/fine_tuned_comparator_full_600_exp016_rescore')],
 'exp-015':[('experiments/exp-015/metrics.json','/prompting_arm_full_600')],
 'exp-026':[('experiments/exp-026/metrics.json','/eval')],
}
for a in ['020','021','022','023','024','025','027','028']:
    m=load(f'{M}/experiments/exp-{a}/metrics.json')
    key=[k for k in m['eval'] if k.startswith('decomposition_report_card')][0]
    SRC[f'exp-{a}']=[(f'experiments/exp-{a}/metrics.json','/eval/'+key),(f'experiments/exp-{a}/metrics.json','/eval/other_reported')]
HOPSRC={}
for a in ['020','021','022','023','024','025','027','028']:
    HOPSRC[f'exp-{a}']=(f'experiments/exp-{a}/metrics.json','/eval/per_gold_hop_report_card')
HOPSRC['exp-026']=('experiments/exp-026/metrics.json','/eval/per_gold_hop_metrics')
HOPSRC['exp-008']=('experiments/exp-008/metrics.json','/eval_metrics/per_gold_hop_metrics')
def run():
    res={}
    for arm,srcs in SRC.items():
        new=load(f'{M}/runs/exp-032/{arm}/eval/eval_metrics.json')
        nf=flat(new)
        rows=[]
        for f,p in srcs:
            try: old=get(load(f'{M}/{f}'),p)
            except KeyError as e: rows.append((f,p,'MISSING',str(e))); continue
            of=flat(old)
            for k,v in of.items():
                if k in nf: rows.append((f+p,k,v,nf[k]))
        hs=HOPSRC.get(arm)
        if hs:
            oh=get(load(f'{M}/{hs[0]}'),hs[1]); nh=new['per_gold_hop_metrics']
            nhr=new.get('per_gold_hop_report_card') or {}
            for h,od in oh.items():
                if not isinstance(od,dict): continue
                nd={**flat(nh.get(h,{})),**flat(nhr.get(h,{}) if isinstance(nhr,dict) else {})}
                for k,v in flat(od).items():
                    if k in nd: rows.append((hs[0]+hs[1]+'/'+h,k,v,nd[k]))
        res[arm]=rows
    summary={}
    for arm,rows in res.items():
        ok=[r for r in rows if r[2]!='MISSING']
        mism=[r for r in ok if not math.isclose(r[2],r[3],rel_tol=0,abs_tol=1e-12)]
        summary[arm]={"values_compared":len(ok),"report_card_values_compared":len([r for r in ok if r[1] in CARD]),
                      "mismatches":len(mism),"report_card_mismatches":len([r for r in mism if r[1] in CARD]),
                      "sources":sorted({r[0].split('/2')[0].split('/3')[0].split('/4')[0] for r in ok}),
                      "expected_identical": arm not in ("exp-015","exp-020","exp-023")}
    return summary

if __name__=="__main__":
    print(json.dumps(run(),indent=1))
