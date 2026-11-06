#!/usr/bin/env python3
"""Convert the pinned encoder plus its shared RVQ encoder weights, strict F32."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'reference'), str(ROOT/'.python-deps')]
from encoder_reference import load_encoder, runtime_tensors, digest, HEARTLIB_REVISION, ENCODER_REVISION, DECODER_REVISION
from gguf import GGUFWriter

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--heartlib',type=Path,required=True)
    p.add_argument('--encoder-snapshot',type=Path,required=True)
    p.add_argument('--decoder-snapshot',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('Existing encoder GGUF is preserved')
    torch.set_num_threads(8)
    encoder,quantizer,config=load_encoder(a.heartlib,a.encoder_snapshot,a.decoder_snapshot)
    tensors,mapping=runtime_tensors(encoder,quantizer)
    # Frozen CPU-built anti-alias kernels keep common DAW rates reproducible.
    import torchaudio
    rates=(16000,22050,32000,44100,48000,88200,96000,192000)
    for rate in rates:
        # tokenize() uses functional.resample with waveform.dtype=F32, whose
        # kernel is constructed in F32. The cached internal 24k->16k transform
        # instead constructs in F64 then casts; preserve that separate buffer.
        tensors[f'rsin.{rate}.kernel']=torchaudio.transforms.Resample(rate,24000,dtype=torch.float32).kernel.contiguous()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    writer=GGUFWriter(str(a.output),'heartcodec-encoder')
    writer.add_uint32('heartcodec.encoder.format_version',1)
    writer.add_string('heartcodec.encoder.source_revision',HEARTLIB_REVISION)
    writer.add_string('heartcodec.encoder.checkpoint_revision',ENCODER_REVISION)
    writer.add_string('heartcodec.encoder.rvq_checkpoint_revision',DECODER_REVISION)
    writer.add_uint32('heartcodec.encoder.chunk_samples',714240)
    writer.add_uint32('heartcodec.encoder.query_dim',512)
    writer.add_uint32('heartcodec.encoder.query_layers',6)
    writer.add_uint32('heartcodec.encoder.query_head_dim',128)
    writer.add_uint32('heartcodec.encoder.query_interval',2)
    writer.add_string('heartcodec.encoder.weight_layout','pytorch-effective-f32-v1')
    writer.add_string('heartcodec.encoder.input_resample_policy','torch-functional-f32-kernel')
    shapes={}
    for name,value in sorted(tensors.items()):
        value=np.ascontiguousarray(value.detach().float().cpu().numpy())
        if not np.isfinite(value).all():raise ValueError('Nonfinite tensor: '+name)
        writer.add_tensor(name,value);shapes[name]=list(value.shape)
    writer.write_header_to_file();writer.write_kv_data_to_file();writer.write_tensors_to_file();writer.close()
    report=dict(heartlib_revision=HEARTLIB_REVISION,encoder_revision=ENCODER_REVISION,decoder_revision=DECODER_REVISION,
                dtype='float32',runtime_tensors=shapes,name_mapping=mapping,gguf_sha256=digest(a.output),
                source_config=config,effective_weight_norm_materialized=True,rvq='Original decoder project_in and eight original codebooks; no new quantizer',
                omitted='BatchNorm training counters, weight-norm source parametrizations after materializing weights, RVQ training buffers and project_out; encoding is strict F32',
                input_rates=(24000,)+rates)
    a.output.with_suffix('.conversion.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Wrote',a.output,len(shapes),'F32 tensors',report['gguf_sha256'],flush=True)

if __name__=='__main__':main()
