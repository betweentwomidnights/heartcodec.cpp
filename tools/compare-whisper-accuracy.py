#!/usr/bin/env python3
"""Candidate Whisper accuracy contract; preserves the original flat-gate verdict.

Predeclared rule: error against the F64 control may be at most twice the larger
upstream CPU/CUDA F32 error (with a 1e-3 absolute floor), AND relative L2 <=1e-4.
This is a review proposal, not an RVQ code-mismatch or legacy-gate waiver.
"""
import argparse,importlib.util,json
from pathlib import Path
import numpy as np

spec=importlib.util.spec_from_file_location('whisper_compare',Path(__file__).with_name('compare-whisper.py'))
compare=importlib.util.module_from_spec(spec);spec.loader.exec_module(compare)
metrics=compare.encoder_compare.metrics
digest=compare.encoder_compare.digest

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('reference-cpu','reference-cuda','reference-f64','native','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Existing report is preserved')
    roots=[a.reference_cpu,a.reference_cuda,a.reference_f64]
    manifests=[json.loads((r/'manifest.json').read_text()) for r in roots]
    if len({m['input_mel_sha256'] for m in manifests})!=1 or len({m['reference_manifest_sha256'] for m in manifests})!=1:
        raise ValueError('Controls must use the same frozen input and checkpoint provenance')
    if [m['device'] for m in manifests]!=['cpu','cuda','cpu'] or [m.get('compute_dtype','f32') for m in manifests]!=['f32','f32','f64']:
        raise ValueError('Expected independent CPU F32, CUDA F32 and CPU F64 controls')
    stages=[]
    for key,meta in manifests[2]['files'].items():
        if key=='output':continue
        controls=[]
        for root,manifest in zip(roots,manifests):
            entry=manifest['files'][key];path=root/entry['file']
            if entry['shape']!=meta['shape'] or digest(path)!=entry['sha256']:raise ValueError('Changed control: '+str(path))
            values=np.fromfile(path,dtype='<f4')
            if not np.isfinite(values).all():raise ValueError('Nonfinite control: '+str(path))
            controls.append(values)
        name=compare.native_name(key);paths=[a.native/(name+'.f32')]
        if not paths[0].exists():paths=[a.native/f'channel-{i}'/(name+'.f32') for i in range(2)]
        if not all(p.exists() for p in paths):
            stages.append(dict(tensor=key,missing=True,candidate_passed=False));continue
        actual=np.concatenate([np.fromfile(p,dtype='<f4') for p in paths])
        cpu,cuda,truth=controls
        cpu_error=metrics(truth,cpu);cuda_error=metrics(truth,cuda);native_error=metrics(truth,actual)
        budget=max(1e-3,2*max(cpu_error['max_abs'],cuda_error['max_abs']))
        valid=native_error.get('shape_match',True) and native_error.get('finite',True)
        entry=dict(tensor=key,upstream_cpu_vs_f64=cpu_error,upstream_cuda_vs_f64=cuda_error,native_vs_f64=native_error,
                   max_abs_budget=budget,legacy_vs_cpu=metrics(cpu,actual),legacy_vs_cuda=metrics(cuda,actual),
                   candidate_passed=bool(valid and native_error['max_abs']<=budget and native_error['relative_l2']<=1e-4))
        stages.append(entry)
    if not stages:raise ValueError('No compared Whisper tensors')
    report=dict(contract='whisper-upstream-envelope-v1-candidate',status='Proposal for review; original gates remain authoritative',
                rule=dict(absolute_floor=1e-3,upstream_error_multiplier=2,relative_l2=1e-4),
                controls=[dict(device=m['device'],compute_dtype=m.get('compute_dtype','f32'),manifest_sha256=digest(root/'manifest.json')) for root,m in zip(roots,manifests)],
                input_mel_sha256=manifests[0]['input_mel_sha256'],reference_case=manifests[0]['case'],stages=stages,
                candidate_passed=all(s['candidate_passed'] for s in stages),
                legacy_cpu_flat_gates_passed=all(s.get('legacy_vs_cpu',{}).get('passed',False) for s in stages),
                legacy_cuda_flat_gates_passed=all(s.get('legacy_vs_cuda',{}).get('passed',False) for s in stages),
                rvq_code_mismatch_waiver=False)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(candidate_passed=report['candidate_passed'],legacy_cpu_flat_gates_passed=report['legacy_cpu_flat_gates_passed'],failed_tensors=[s['tensor'] for s in stages if not s['candidate_passed']]),indent=2))
    raise SystemExit(0 if report['candidate_passed'] else 1)

if __name__=='__main__':main()
