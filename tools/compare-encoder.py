#!/usr/bin/env python3
"""Offline FP32 encoder stage/code comparison; no inference or Torch dependency."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(16*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def metrics(a,b):
    if a.shape!=b.shape or not np.isfinite(b).all():return dict(shape_match=a.shape==b.shape,finite=bool(np.isfinite(b).all()),passed=False)
    x,y=a.astype(np.float64),b.astype(np.float64);delta=y-x
    maximum=float(np.max(np.abs(delta))) if delta.size else 0
    relative=float(np.linalg.norm(delta)/max(float(np.linalg.norm(x)),1e-30))
    return dict(max_abs=maximum,relative_l2=relative,passed=maximum<=1e-3 and relative<=1e-4)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--native',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=('raw','query','codes'),default='raw')
    p.add_argument('--require-matching-backend',action='store_true',help='Reject missing or different native/reference backend labels')
    p.add_argument('--case',default='source-2976')
    a=p.parse_args()
    if a.output.exists():raise ValueError('Existing comparison report is preserved')
    manifest=json.loads((a.reference/'manifest.json').read_text())
    native_run=json.loads((a.native/'run.json').read_text())
    backend_match=(str(native_run['backend']).lower()==str(manifest['device']).lower()) if native_run.get('backend') else None
    if a.require_matching_backend and backend_match is not True:raise ValueError('Native backend must be recorded and match the reference device')
    cases=[c for c in manifest['cases'] if c['name'].startswith('source')] if a.mode in ('raw','codes') else [c for c in manifest['cases'] if c['name']==a.case]
    if not cases:raise ValueError('No matching reference cases')
    stages=[];margin_diagnostics=[];reference_codes=[]
    for index,case in enumerate(cases):
        root=a.reference/case['name'];native=a.native/'stages'/f'chunk-{index:06d}' if a.mode=='raw' else a.native
        for key,spec in case['files'].items():
            if a.mode=='codes' and key!='codes':continue
            path=root/spec['file']
            if digest(path)!=spec['sha256']:raise ValueError('Changed golden: '+str(path))
            if key=='codes':reference_codes.append(np.fromfile(path,dtype='<i4').reshape(-1,8));continue
            if a.mode=='query' and not (key=='query' or key=='rvq.projected' or key.startswith(('stage.query.','rvq.margin.'))):continue
            # Quantized 512-D decode is provided by the existing flow capability,
            # not the new encoding API. Waveform is a borrowed input fixture.
            if key in ('waveform','rvq.quantized'):continue
            expected=np.fromfile(path,dtype='<f4')
            output=(a.native/'stages'/spec['file']) if key=='resampled24' and a.mode=='raw' else native/spec['file']
            if output.exists():actual=np.fromfile(output,dtype='<f4')
            elif (native/'channel-0'/spec['file']).exists() and (native/'channel-1'/spec['file']).exists():
                actual=np.concatenate([np.fromfile(native/f'channel-{ch}'/spec['file'],dtype='<f4') for ch in range(2)])
            else:
                stages.append(dict(case=case['name'],tensor=key,missing=True,passed=False));continue
            metric=metrics(expected,actual);entry=dict(case=case['name'],tensor=key,**metric)
            if key.startswith('rvq.margin.'):
                entry.pop('passed',None);margin_diagnostics.append(entry)
            else:stages.append(entry)
    expected=np.concatenate(reference_codes)
    if a.mode in ('raw','codes'):expected=expected[:manifest.get('source_complete_frames',len(expected))]
    actual=np.fromfile(a.native/'codes.i32',dtype='<i4').reshape(-1,8)
    code_shape=actual.shape==expected.shape
    changes=np.argwhere(expected!=actual) if code_shape else np.empty((0,2),dtype=np.int64)
    code_passed=code_shape and not len(changes)
    code_differences=[dict(frame=int(f),level=int(q),expected=int(expected[f,q]),actual=int(actual[f,q])) for f,q in changes[:128]]
    report=dict(reference_manifest_sha256=digest(a.reference/'manifest.json'),model_sha256=digest(a.model),
                reference_heartlib_revision=manifest['heartlib_revision'],reference_encoder_revision=manifest['encoder_revision'],
                reference_decoder_revision=manifest['decoder_revision'],reference_device=manifest['device'],
                native_run=native_run,mode=a.mode,scope='code IDs only; activations not evaluated' if a.mode=='codes' else 'activation stages and code IDs',
                backend_match=backend_match,matching_backend_required=a.require_matching_backend,
                gates=dict(code_ids='all exact') if a.mode=='codes' else dict(max_abs=1e-3,relative_l2=1e-4,code_ids='all exact',margin_values='diagnostic only; not a gate waiver'),
                stages=stages,rvq_margin_diagnostics=margin_diagnostics,
                codes=dict(shape_match=code_shape,expected_frames=len(expected),actual_frames=len(actual),
                           total_compared_codes=int(expected.size),changed_codes=int(len(changes)) if code_shape else None,
                           differences=code_differences,passed=bool(code_passed),sha256=digest(a.native/'codes.i32')),
                skipped_reference_values=['waveform: borrowed PCM input','query mode: borrowed feature inputs and upstream audio/feature stages outside the tested query boundary','rvq.quantized: decoder projection, outside encoding API'],
                all_stage_gates_passed=None if a.mode=='codes' else all(s['passed'] for s in stages),code_parity_passed=bool(code_passed),
                passed=all(s['passed'] for s in stages) and code_passed)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    failures=[s['tensor'] for s in stages if not s['passed']]
    print(json.dumps(dict(passed=report['passed'],changed_codes=report['codes']['changed_codes'],failed_stages=failures),indent=2))
    raise SystemExit(0 if report['passed'] else 1)

if __name__=='__main__':main()
