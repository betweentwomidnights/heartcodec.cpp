#!/usr/bin/env python3
"""Freeze stock Whisper stages on one existing, hash-checked encoder mel input."""
import argparse,json,os
from pathlib import Path
import numpy as np
import torch
from encoder_reference import load_encoder,digest,HEARTLIB_REVISION,ENCODER_REVISION,DECODER_REVISION

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('heartlib','encoder-snapshot','decoder-snapshot','reference','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--case',default='source-chunk-0')
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--dtype',choices=('f32','f64'),default='f32',help='F64 is a numerical diagnostic, not a replacement golden')
    p.add_argument('--threads',type=int,default=8)
    p.add_argument('--disable-mkldnn',action='store_true',help='Backend diagnostic; preserves the original oracle separately')
    p.add_argument('--linear-control',action='store_true',help='Also audit layer6 FC2 using its exact frozen activated input')
    a=p.parse_args()
    if a.output.exists():raise ValueError('Use a fresh Whisper reference directory')
    if a.linear_control and a.dtype!='f32':raise ValueError('Projection control requires the original F32 model')
    torch.set_num_threads(a.threads);torch.manual_seed(1701)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    if a.disable_mkldnn:torch.backends.mkldnn.enabled=False
    encoder,_,_=load_encoder(a.heartlib,a.encoder_snapshot,a.decoder_snapshot)
    dtype=torch.float64 if a.dtype=='f64' else torch.float32
    model=encoder.feature_extractors.whisper_encoder.to(device=a.device,dtype=dtype)
    manifest=json.loads((a.reference/'manifest.json').read_text())
    case=next(c for c in manifest['cases'] if c['name']==a.case)
    spec=case['files']['input.whisper_mel'];path=a.reference/a.case/spec['file']
    if digest(path)!=spec['sha256']:raise ValueError('Changed frozen mel input')
    x=torch.from_numpy(np.fromfile(path,dtype='<f4').copy().reshape(spec['shape'])).to(device=a.device,dtype=dtype)
    a.output.mkdir(parents=True);files={}
    def save(key,value):
        value=value.detach().float().cpu().numpy();value=np.ascontiguousarray(value)
        path=a.output/(key+'.f32');value.tofile(path)
        files[key]=dict(file=path.name,shape=list(value.shape),sha256=digest(path))
    handles=[]
    for name,child in model.named_modules():
        if name in ('conv1','conv2','layer_norm') or name.startswith('layers.') and (name.count('.')==1 or name.startswith(('layers.0.','layers.5.','layers.6.','layers.11.'))):
            def capture(m,i,o,name=name):
                value=o[0] if isinstance(o,tuple) else o
                save('trace.whisper.'+name,value)
                if name=='conv1':save('trace.whisper.gelu1',torch.nn.functional.gelu(value).transpose(1,2))
                if name=='conv2':save('trace.whisper.embedded',torch.nn.functional.gelu(value).transpose(1,2)+model.embed_positions.weight)
                if name.endswith('.fc1'):save('trace.whisper.'+name.rsplit('.',1)[0]+'.activated',model.layers[int(name.split('.')[1])].activation_fn(value))
            handles.append(child.register_forward_hook(capture))
            if name.endswith('.self_attn.out_proj'):
                handles.append(child.register_forward_pre_hook(lambda m,i,name=name:save('trace.whisper.'+name.rsplit('.',2)[0]+'.attn_head_output',i[0])))
    with torch.inference_mode():output=model(x,return_dict=True).last_hidden_state
    for h in handles:h.remove()
    save('output',output)
    report=dict(reference_manifest_sha256=digest(a.reference/'manifest.json'),case=a.case,device=a.device,
                heartlib_revision=HEARTLIB_REVISION,encoder_revision=ENCODER_REVISION,decoder_revision=DECODER_REVISION,
                torch=torch.__version__,threads=a.threads,compute_dtype=a.dtype,stored_dtype='f32',attention_implementation=model.config._attn_implementation,
                mkldnn_enabled=torch.backends.mkldnn.enabled,float32_matmul_precision=torch.get_float32_matmul_precision(),
                precision_environment={k:os.environ.get(k) for k in ('ONEDNN_DEFAULT_FPMATH_MODE','DNNL_DEFAULT_FPMATH_MODE','NVIDIA_TF32_OVERRIDE')},
                exporter_sha256=digest(__file__),input_mel_sha256=spec['sha256'],files=files)
    if a.linear_control:
        block=model.layers[6].fc2
        activation_path=a.output/'trace.whisper.layers.6.activated.f32'
        activated=torch.from_numpy(np.fromfile(activation_path,dtype='<f4').copy().reshape(-1,3072)).to(a.device)
        frozen=torch.from_numpy(np.fromfile(a.output/'trace.whisper.layers.6.fc2.f32',dtype='<f4').copy().reshape(-1,768)).to(a.device)
        def metric(x,y):
            delta=x.double()-y.double()
            return dict(max_abs=float(delta.abs().max()),relative_l2=float(delta.norm()/y.double().norm()))
        with torch.inference_mode():
            linear=block(activated);separate=activated@block.weight.T+block.bias
            truth=torch.nn.functional.linear(activated.double(),block.weight.double(),block.bias.double()).float()
        report['linear6_control']=dict(input_sha256=digest(activation_path),repeated_linear_vs_frozen=metric(linear,frozen),
                                       linear_vs_f64=metric(linear,truth),separate_matmul_bias_vs_f64=metric(separate,truth),
                                       separate_matmul_bias_vs_linear=metric(separate,linear))
        control_path=a.output/'control.linear6-f64.f32'
        np.ascontiguousarray(truth.detach().cpu().numpy()).tofile(control_path)
        report['linear6_control']['f64_output']=dict(file=control_path.name,shape=list(truth.shape),sha256=digest(control_path))
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='files'}),flush=True)

if __name__=='__main__':main()
