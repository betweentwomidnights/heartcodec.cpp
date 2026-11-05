#!/usr/bin/env python3
"""Query precision controls on frozen features; F64 modes are diagnostics only."""
import argparse,json,types
from pathlib import Path
import numpy as np
import torch
from torch.nn.utils import parametrize
from encoder_reference import load_encoder,digest,HEARTLIB_REVISION,ENCODER_REVISION,DECODER_REVISION

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('heartlib','encoder-snapshot','decoder-snapshot','reference','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--case',default='source-chunk-0')
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--arithmetic',choices=('source-f32','source-mixed-f64','diagnostic-f64'),default='source-f32')
    a=p.parse_args()
    if a.output.exists():raise ValueError('Use a fresh query control directory')
    torch.set_num_threads(8);torch.manual_seed(1701)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    encoder,quantizer,_=load_encoder(a.heartlib,a.encoder_snapshot,a.decoder_snapshot)
    model=encoder.query
    for child in list(model.modules()):
        if hasattr(child,'parametrizations') and 'weight' in child.parametrizations:parametrize.remove_parametrizations(child,'weight',leave_parametrized=True)
    dtype=torch.float32 if a.arithmetic=='source-f32' else torch.float64
    model.to(device=a.device,dtype=dtype);quantizer.to(device=a.device,dtype=dtype)
    if a.arithmetic=='diagnostic-f64':
        def rotate(x,frequencies):
            width=frequencies.shape[-1];first,second=x[...,:width].chunk(2,dim=-1)
            rotated=x[...,:width]*frequencies.cos()+torch.cat((-second,first),dim=-1)*frequencies.sin()
            return torch.cat((rotated,x[...,width:]),dim=-1)
        model.encoder_transformers[0].self_attn.forward.__globals__['_apply_rotary']=rotate
        for block in model.encoder_transformers:
            def frequency(self,length):
                positions=torch.arange(length,device=self.inv_freq.device,dtype=torch.float64)
                f=positions[:,None]*self.inv_freq.double()[None,:]
                return torch.cat((f,f),dim=-1)
            block.rope.forward=types.MethodType(frequency,block.rope)
    manifest=json.loads((a.reference/'manifest.json').read_text());case=next(c for c in manifest['cases'] if c['name']==a.case)
    inputs=[];hashes={}
    for key in ('acoustic','semantic','whisper','wavlm'):
        entry=case['files']['input.'+key];path=a.reference/a.case/entry['file']
        if digest(path)!=entry['sha256']:raise ValueError('Changed feature input: '+str(path))
        inputs.append(torch.from_numpy(np.fromfile(path,dtype='<f4').copy().reshape(entry['shape'])).to(device=a.device,dtype=dtype));hashes[key]=entry['sha256']
    a.output.mkdir(parents=True);files={}
    def save(key,value):
        array=np.ascontiguousarray(value.detach().cpu().numpy());path=a.output/(key+('.i32' if array.dtype==np.int32 else '.f64' if array.dtype==np.float64 else '.f32'))
        array.tofile(path);files[key]=dict(file=path.name,shape=list(array.shape),dtype=str(array.dtype),sha256=digest(path))
    hooks=[]
    for i,block in enumerate(model.encoder_transformers):hooks.append(block.register_forward_hook(lambda m,i,o,layer=i:save('layer.'+str(layer),o)))
    with torch.inference_mode():
        query=model(*inputs);save('query',query);projected=quantizer.project_in(query);save('projected',projected)
        if a.arithmetic=='source-f32':
            _,codes,_=quantizer(query);save('source.codes',codes[0].to(torch.int32))
        residual=projected.double();codes=[];margins=[]
        for layer in quantizer.layers:
            embed=layer._codebook.embed[0].double()
            distances=((residual.unsqueeze(-2)-embed)**2).sum(-1)
            top=torch.topk(distances,2,dim=-1,largest=False);selected=distances.argmin(-1)
            codes.append(selected);margins.append(top.values[...,1]-top.values[...,0])
            residual=residual-torch.nn.functional.embedding(selected,embed)
        save('direct_f64.codes',torch.stack(codes,-1)[0].to(torch.int32));save('direct_f64.margins',torch.stack(margins,-1)[0])
    for h in hooks:h.remove()
    report=dict(heartlib_revision=HEARTLIB_REVISION,encoder_revision=ENCODER_REVISION,decoder_revision=DECODER_REVISION,
                reference_manifest_sha256=digest(a.reference/'manifest.json'),case=a.case,device=a.device,arithmetic=a.arithmetic,
                rotary_policy='Diagnostic F64 angles and rotation; source F32 casts deliberately removed' if a.arithmetic=='diagnostic-f64' else 'Original source F32 rotary policy retained',
                effective_weights_materialized_on_cpu=True,input_feature_sha256=hashes,files=files,exporter_sha256=digest(__file__))
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    frame=min(259,query.shape[1]-1)
    print(a.arithmetic,a.device,'frame',frame,'direct F64 codes',torch.stack(codes,-1)[0,frame].tolist(),flush=True)

if __name__=='__main__':main()
