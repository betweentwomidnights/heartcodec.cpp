#!/usr/bin/env python3
"""Compare frozen Whisper traces without changing encoder parity tolerances."""
import argparse,json,importlib.util
from pathlib import Path
import numpy as np
spec=importlib.util.spec_from_file_location('encoder_compare',Path(__file__).with_name('compare-encoder.py'))
encoder_compare=importlib.util.module_from_spec(spec);spec.loader.exec_module(encoder_compare)

def native_name(key):
    if key in ('trace.whisper.conv1','trace.whisper.conv2'):
        return key.replace('trace.whisper','stage.feature.whisper_encoder')
    if key=='trace.whisper.layer_norm':return 'stage.feature.whisper_encoder'
    if key.startswith('trace.whisper.layers.'):
        if key.count('.')==3:return key.replace('trace.whisper','stage.feature.whisper_encoder')
        if key.endswith('.self_attn'):return key+'.out_proj'
        if key.endswith('.activation_fn'):return key.rsplit('.',1)[0]+'.activated'
    return key

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--native',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--layer',type=int,choices=range(12))
    p.add_argument('--projection',choices=('fc1','fc2'))
    a=p.parse_args()
    if a.projection and a.layer is None:p.error('--projection requires --layer')
    if a.output.exists():raise ValueError('Existing report is preserved')
    manifest=json.loads((a.reference/'manifest.json').read_text());stages=[]
    for key,spec in manifest['files'].items():
        if key=='output':continue # Same final tensor as layer_norm.
        if a.layer is not None and not (key==f'trace.whisper.layers.{a.layer}' or key.startswith(f'trace.whisper.layers.{a.layer}.')):continue
        if a.projection and key!=f'trace.whisper.layers.{a.layer}.{a.projection}':continue
        expected_path=a.reference/spec['file']
        if encoder_compare.digest(expected_path)!=spec['sha256']:raise ValueError('Changed golden: '+str(expected_path))
        name=native_name(key)
        paths=[a.native/(name+'.f32')]
        if not paths[0].exists():paths=[a.native/f'channel-{i}'/(name+'.f32') for i in range(2)]
        # Also support an independently exported Torch control directory.
        if not all(p.exists() for p in paths) and (a.native/spec['file']).exists():paths=[a.native/spec['file']]
        if not all(p.exists() for p in paths):
            stages.append(dict(tensor=key,missing=True,passed=False));continue
        expected=np.fromfile(expected_path,dtype='<f4');actual=np.concatenate([np.fromfile(p,dtype='<f4') for p in paths])
        metric=encoder_compare.metrics(expected,actual);entry=dict(tensor=key,**metric)
        if expected.shape==actual.shape and len(expected):
            i=int(np.argmax(np.abs(actual.astype('f8')-expected.astype('f8'))))
            entry.update(max_location=[int(x) for x in np.unravel_index(i,spec['shape'])],reference_at_max=float(expected[i]),native_at_max=float(actual[i]),reference_peak=float(np.max(np.abs(expected))))
        stages.append(entry)
    if not stages:raise ValueError('No compared Whisper stages')
    report=dict(reference_manifest_sha256=encoder_compare.digest(a.reference/'manifest.json'),reference_case=manifest['case'],
                reference_device=manifest['device'],reference_compute_dtype=manifest.get('compute_dtype','f32'),
                input_mel_sha256=manifest['input_mel_sha256'],layer=a.layer,
                gates=dict(max_abs=1e-3,relative_l2=1e-4),stages=stages,passed=all(s['passed'] for s in stages))
    if a.layer is not None:
        boundary=f'trace.whisper.layers.{a.layer-1}' if a.layer else 'trace.whisper.embedded'
        if a.projection:boundary=f'trace.whisper.layers.{a.layer}.'+('activated' if a.projection=='fc2' else 'final_layer_norm')
        report['reference_hidden_input']=dict(tensor=boundary,**manifest['files'][boundary])
    if (a.native/'run.json').exists():report['native_run']=json.loads((a.native/'run.json').read_text())
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=report['passed'],failed_tensors=[s['tensor'] for s in stages if not s['passed']]),indent=2))
    raise SystemExit(0 if report['passed'] else 1)

if __name__=='__main__':main()
