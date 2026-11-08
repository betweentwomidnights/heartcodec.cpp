#!/usr/bin/env python3
"""Freeze encoder audit, feature-to-query/RVQ stages and optional full audio stages."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from encoder_reference import load_encoder, runtime_tensors, digest, HEARTLIB_REVISION, ENCODER_REVISION, DECODER_REVISION

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--heartlib', type=Path, required=True)
    p.add_argument('--encoder-snapshot', type=Path, required=True)
    p.add_argument('--decoder-snapshot', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--feature-frames', default='8,24,744')
    p.add_argument('--audio', type=Path)
    p.add_argument('--expected-codes', type=Path)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Use a fresh encoder fixture directory')
    torch.set_num_threads(8); torch.manual_seed(1701)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    encoder, quantizer, config = load_encoder(a.heartlib, a.encoder_snapshot, a.decoder_snapshot)
    tensors, mapping = runtime_tensors(encoder, quantizer)
    a.output.mkdir(parents=True)
    audit = dict(heartlib_revision=HEARTLIB_REVISION, encoder_revision=ENCODER_REVISION,
                 decoder_revision=DECODER_REVISION, config=config, source_tensor_count=len(encoder.state_dict()),
                 runtime_tensors={name: list(value.shape) for name, value in tensors.items()},
                 runtime_name_mapping=mapping, runtime_f32_bytes=sum(t.numel()*4 for t in tensors.values()))
    (a.output/'audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    del tensors
    report = dict(heartlib_revision=HEARTLIB_REVISION, encoder_revision=ENCODER_REVISION,
                  decoder_revision=DECODER_REVISION, torch=torch.__version__, numpy=np.__version__,
                  dtype='float32', device='cpu', threads=8, exporter_sha256=digest(__file__),
                  loader_sha256=digest(Path(__file__).with_name('encoder_reference.py')), cases=[])
    def case(name, features, audio=None):
        root = a.output/name; root.mkdir()
        files = {}
        def save(key, value):
            value = np.ascontiguousarray(value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value)
            filename = key + ('.i32' if value.dtype==np.int32 else '.f32')
            path = root/filename; value.tofile(path)
            files[key] = dict(file=filename, shape=list(value.shape), dtype=str(value.dtype), sha256=digest(path))
        for key, value in zip(('acoustic', 'semantic', 'whisper', 'wavlm'), features):
            save('input.'+key, value)
        if audio is not None:
            for key, value in audio.items(): save(key,value)
        handles=[]
        for key, child in encoder.query.named_modules():
            if key in ('cond_fusion_layer_acoustic','cond_fusion_layer_semantic','cond_fusion_layer_phone',
                       'feature_proj','encoder_transformers.0','encoder_transformers.1','encoder_transformers.5'):
                handles.append(child.register_forward_hook(lambda m,i,o,key=key:save('stage.query.'+key,o)))
        projected=[]
        handles.append(quantizer.project_in.register_forward_hook(lambda m,i,o:projected.append(o.detach().clone())))
        with torch.inference_mode():
            output=encoder.query(*features);save('query',output)
            quantized,codes,_=quantizer(output);save('codes',codes[0].to(torch.int32))
            save('rvq.projected',projected[0]);save('rvq.quantized',quantized)
            # Save exact score margins for boundary diagnostics; never loosen gates.
            residual=projected[0].clone()
            for level, layer in enumerate(quantizer.layers):
                embed=layer._codebook.embed[0]
                from vector_quantize_pytorch.vector_quantize_pytorch import cdist
                distance=-cdist(residual,embed.unsqueeze(0))
                top=torch.topk(distance,2,dim=-1)
                save(f'rvq.margin.{level}',top.values[...,0]-top.values[...,1])
                residual=residual-torch.nn.functional.embedding(codes[...,level],embed)
        for handle in handles:handle.remove()
        report['cases'].append(dict(name=name,feature_frames=int(features[0].shape[1]),query_frames=int(output.shape[1]),files=files))
        print('Frozen',name,output.shape,codes.shape,flush=True)
        return codes[0]
    for frames in map(int,a.feature_frames.split(',')):
        if frames<=0 or frames%2:raise ValueError('Feature frames must be positive and even')
        features=tuple(torch.randn(1,frames,d) for d in (2048,2048,1536,1536))
        case('synthetic-'+str(frames),features)
    if a.audio:
        import soundfile as sf
        from torchaudio.functional import resample
        waveform,rate=sf.read(a.audio,dtype='float32',always_2d=True)
        if waveform.shape[1] not in (1,2):raise ValueError('Expected mono/stereo audio')
        audio=torch.from_numpy(waveform.T.copy())
        if audio.shape[0]==1:audio=audio.expand(2,-1)
        original=audio
        if rate!=24000:audio=resample(audio,rate,24000)
        all_codes=[]
        for chunk_index,first in enumerate(range(0,audio.shape[-1],714240)):
            piece=audio[:,first:first+714240]
            chunk=torch.nn.functional.pad(piece,(0,714480-piece.shape[-1])).unsqueeze(0)
            values={'chunk24':chunk}
            if chunk_index==0:values.update(waveform=original,resampled24=audio)
            handles=[]
            fe=encoder.feature_extractors
            handles.append(fe.whisper_encoder.register_forward_pre_hook(lambda m,i:values.__setitem__('input.whisper_mel',i[0].detach().cpu())))
            handles.append(fe.muencoder.conv.register_forward_pre_hook(lambda m,i:values.__setitem__('input.mu_mel',i[0].detach().cpu())))
            for name,child in fe.named_modules():
                if name in ('resample_24k_to_16k','muencoder.preprocessor_melspec_2048',
                            'muencoder.conv','whisper_encoder.conv1','whisper_encoder.conv2',
                            'whisper_encoder','whisper_encoder.layers.0','whisper_encoder.layers.11',
                            'muencoder.conformer.layers.0','muencoder.conformer.layers.1','muencoder.conformer.layers.10',
                            'wavlm_encoder.feature_extractor','wavlm_encoder.feature_projection',
                            'wavlm_encoder.encoder.pos_conv_embed','wavlm_encoder.encoder.layers.0',
                            'wavlm_encoder.encoder.layers.5','wavlm_encoder.encoder.layers.8'):
                    def capture(m,i,o,name=name):
                        value=o.last_hidden_state if hasattr(o,'last_hidden_state') else o[0] if isinstance(o,tuple) else o
                        values['stage.feature.'+name]=value.detach().cpu()
                    handles.append(child.register_forward_hook(capture))
            with torch.inference_mode():features=fe(chunk)
            for h in handles:h.remove()
            name='source-2976' if audio.shape[-1]==714240 else 'source-chunk-'+str(chunk_index)
            all_codes.append(case(name,features,values))
        complete_frames=len(waveform)*25//(rate*2)
        codes=torch.cat(all_codes)[:complete_frames]
        report['source_complete_frames']=complete_frames
        if a.expected_codes:
            expected=np.fromfile(a.expected_codes,dtype='<i4').reshape(-1,8)
            if not np.array_equal(expected,codes.numpy()):raise ValueError('Fresh CPU encoder does not reproduce frozen source RVQ codes')
            report['source_codes_match_existing']=True
        report['source_audio_sha256']=digest(a.audio);report['source_sample_rate']=rate
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
