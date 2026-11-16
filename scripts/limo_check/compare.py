import numpy as np,os,json
D=os.path.expanduser("~/training-forensics/results/limo_check/dl/")
L=lambda t: np.fromfile(D+t+".bin",dtype=np.uint16).reshape(152064,5120)
limo=L("limo"); res={}
for b in ["q32i","q32b","qc32i","qc32b","limo2"]:
    if not os.path.exists(D+b+".bin"): continue
    x=L(b); ne=(x!=limo); rows=ne.any(1)
    res[b]={"rows_identical":int((~rows).sum()),"rows_changed":int(rows.sum()),"elems_changed_frac":float(ne.mean())}
    print(b,res[b],flush=True)
    if b=="q32i": np.save(D+"changed_mask.npy",rows); np.save(D+"frac_changed.npy",ne.mean(1))
b=L("q32i"); v2=L("limo2"); r2=(v2!=b).any(1); print("limo2 vs q32i changed",int(r2.sum()),"identical",int((~r2).sum())); np.save(D+"changed_mask_v2.npy",r2)
r1=np.load(D+"changed_mask.npy"); print("v1&v2 changed",int((r1&r2).sum()),"v1 only",int((r1&~r2).sum()),"v2 only",int((~r1&r2).sum()))
# special/added token rows
print("rows>=151643 changed v1:",np.nonzero(r1[151643:])[0]+151643)
json.dump(res,open(D+"../compare_parents.json","w"),indent=1)
