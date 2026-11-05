#!/usr/bin/env python3
"""Audit an RVQ decision using F64 distances; never alters runtime code selection."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'.python-deps'))
from gguf import GGUFReader

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(16*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--sample',nargs=3,action='append',required=True,metavar=('LABEL','PROJECTED_FILE','CODES_FILE'))
    p.add_argument('--frame',type=int,default=259);p.add_argument('--level',type=int,default=4)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Existing boundary report is preserved')
    if a.frame<0 or not 0<=a.level<8:raise ValueError('Invalid frame/level')
    reader=GGUFReader(str(a.model));tensors={t.name:np.asarray(t.data) for t in reader.tensors}
    if not all(tensors[f'rvq.layers.{i}._codebook.embed'].shape==(8192,32) for i in range(8)):raise ValueError('Unexpected RVQ codebook shapes')
    samples=[];first_residual=None;first_prefix=None
    for label,projected_path,codes_path in a.sample:
        projected_path,codes_path=Path(projected_path),Path(codes_path)
        if projected_path.suffix not in ('.f32','.f64'):raise ValueError('Projected file must explicitly identify F32/F64 dtype')
        dtype='<f8' if projected_path.suffix=='.f64' else '<f4'
        projected=np.fromfile(projected_path,dtype=dtype).reshape(-1,32);codes=np.fromfile(codes_path,dtype='<i4').reshape(-1,8)
        if a.frame>=len(projected) or a.frame>=len(codes) or not np.isfinite(projected).all() or np.any((codes<0)|(codes>=8192)):raise ValueError('Invalid projected values/codes or frame')
        residual=projected[a.frame].copy();prefix=codes[a.frame,:a.level]
        for i,code in enumerate(prefix):residual-=tensors[f'rvq.layers.{i}._codebook.embed'][code].astype(dtype)
        residual=residual.astype('f8');embed=tensors[f'rvq.layers.{a.level}._codebook.embed'].astype('f8')
        d2=((embed-residual)**2).sum(axis=1);best=np.argsort(d2,kind='stable')[:5]
        separation=float(np.linalg.norm(embed[best[1]]-embed[best[0]]));margin=float(d2[best[1]]-d2[best[0]])
        if first_residual is None:first_residual=residual;first_prefix=prefix
        same_prefix=bool(np.array_equal(prefix,first_prefix))
        samples.append(dict(label=label,projected_sha256=digest(projected_path),codes_sha256=digest(codes_path),projected_dtype=dtype,
                            residual_policy='Subtract earlier embeddings in the projected dtype, then measure distances in F64',
                            prefix=prefix.tolist(),recorded_code=int(codes[a.frame,a.level]),f64_nearest=int(best[0]),squared_distance_margin=margin,
                            distance_to_nearest_bisector=margin/(2*separation) if separation else 0,
                            prefix_equal_to_first=same_prefix,residual_l2_displacement_from_first=float(np.linalg.norm(residual-first_residual)) if same_prefix else None,
                            top=[dict(code=int(i),squared_distance=float(d2[i])) for i in best]))
    report=dict(model_sha256=digest(a.model),frame=a.frame,level=a.level,scope='Diagnostic only; no tie epsilon, reranking, code remapping or runtime change',samples=samples)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps([dict(label=s['label'],recorded=s['recorded_code'],f64_nearest=s['f64_nearest'],boundary_distance=s['distance_to_nearest_bisector']) for s in samples],indent=2))

if __name__=='__main__':main()
