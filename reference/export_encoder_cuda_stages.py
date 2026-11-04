#!/usr/bin/env python3
"""CUDA neural-stage oracle from frozen CPU frontend inputs and effective weights.

This is not a stock end-to-end CUDA tokenize run: that image's complex STFT
Jiterator path does not support GB10. CPU waveform/STFT fixtures remain separate.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn.utils import parametrize
from encoder_reference import load_encoder,digest,HEARTLIB_REVISION,ENCODER_REVISION,DECODER_REVISION

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('heartlib','encoder-snapshot','decoder-snapshot','reference','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Use a fresh CUDA-stage oracle directory')
    torch.set_num_threads(8);torch.manual_seed(1701)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    encoder,quantizer,config=load_encoder(a.heartlib,a.encoder_snapshot,a.decoder_snapshot)
    # GGUF conversion materializes these exact effective weights on CPU.
    for child in list(encoder.modules()):
        if hasattr(child,'parametrizations') and 'weight' in child.parametrizations:
            parametrize.remove_parametrizations(child,'weight',leave_parametrized=True)
    encoder.query.cuda();quantizer.cuda()
    fe=encoder.feature_extractors
    fe.whisper_encoder.cuda();fe.wavlm_encoder.cuda()
    fe.muencoder.conv.cuda();fe.muencoder.conformer.cuda()
    reference=json.loads((a.reference/'manifest.json').read_text())
    report=dict(heartlib_revision=HEARTLIB_REVISION,encoder_revision=ENCODER_REVISION,decoder_revision=DECODER_REVISION,
                torch=torch.__version__,numpy=np.__version__,dtype='float32',device='cuda',
                device_name=torch.cuda.get_device_name(),capability=torch.cuda.get_device_capability(),
                input_cpu_manifest_sha256=digest(a.reference/'manifest.json'),
                effective_weights_materialized_on_cpu=True,frontend='Frozen CPU tensors; no end-to-end CUDA tokenize claim',cases=[])
    a.output.mkdir(parents=True)
    for case in reference['cases']:
        if not case['name'].startswith('source'):continue
        source=a.reference/case['name'];root=a.output/case['name'];root.mkdir();files={}
        def load(key):
            spec=case['files'][key];path=source/spec['file']
            if digest(path)!=spec['sha256']:raise ValueError('Changed CPU stage input')
            return torch.from_numpy(np.fromfile(path,dtype='<f4').copy().reshape(spec['shape'])).cuda()
        def save(key,value):
            array=np.ascontiguousarray(value.detach().cpu().numpy())
            path=root/(key+('.i32' if array.dtype==np.int32 else '.f32'));array.tofile(path)
            files[key]=dict(file=path.name,shape=list(array.shape),dtype=str(array.dtype),sha256=digest(path))
        handles=[]
        for name,child in fe.named_modules():
            if name in ('muencoder.conv','muencoder.conformer.layers.0','muencoder.conformer.layers.1','muencoder.conformer.layers.10',
                        'whisper_encoder','whisper_encoder.conv1','whisper_encoder.conv2','whisper_encoder.layers.0','whisper_encoder.layers.11',
                        'wavlm_encoder.feature_extractor','wavlm_encoder.feature_projection','wavlm_encoder.encoder.pos_conv_embed',
                        'wavlm_encoder.encoder.layers.0','wavlm_encoder.encoder.layers.5','wavlm_encoder.encoder.layers.8'):
                def capture(m,i,o,name=name):
                    value=o.last_hidden_state if hasattr(o,'last_hidden_state') else o[0] if isinstance(o,tuple) else o
                    save('stage.feature.'+name,value)
                handles.append(child.register_forward_hook(capture))
        with torch.inference_mode():
            mu_mel=load('input.mu_mel');wh_mel=load('input.whisper_mel');audio16=load('stage.feature.resample_24k_to_16k')
            hidden=fe.muencoder.conformer(fe.muencoder.conv(mu_mel),output_hidden_states=True,return_dict=True).hidden_states
            acoustic,semantic=hidden[2],hidden[11]
            states=fe.wavlm_encoder(audio16,output_hidden_states=True,return_dict=True).hidden_states
            wavlm=torch.stack(states[6:10],dim=1).mean(dim=1)
            whisper=fe.whisper_encoder(wh_mel,return_dict=True).last_hidden_state[:,:1488]
            whisper=whisper.reshape(2,744,2,768).mean(dim=2);wavlm=wavlm.reshape(2,744,2,768).mean(dim=2)
            features=tuple(fe._stereo_features(x,1) for x in (acoustic,semantic,whisper,wavlm))
            for key,value in zip(('acoustic','semantic','whisper','wavlm'),features):save('input.'+key,value)
            for handle in handles:handle.remove()
            for name,child in encoder.query.named_modules():
                if name in ('cond_fusion_layer_acoustic','cond_fusion_layer_semantic','cond_fusion_layer_phone','feature_proj',
                            'encoder_transformers.0','encoder_transformers.1','encoder_transformers.5'):
                    handles.append(child.register_forward_hook(lambda m,i,o,name=name:save('stage.query.'+name,o)))
            projected=[]
            handles.append(quantizer.project_in.register_forward_hook(lambda m,i,o:projected.append(o.detach())))
            query=encoder.query(*features);save('query',query)
            _,codes,_=quantizer(query);save('codes',codes[0].to(torch.int32));save('rvq.projected',projected[0])
            for handle in handles:handle.remove()
        report['cases'].append(dict(name=case['name'],feature_frames=744,query_frames=372,files=files))
        print('Frozen CUDA neural stages',case['name'],flush=True)
    report['source_complete_frames']=reference.get('source_complete_frames')
    report['source_audio_sha256']=reference.get('source_audio_sha256')
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
