"""Strict, isolated loading of the pinned waveform encoder and shared decoder RVQ.

Reference/conversion only. Run in the dedicated Spark image, not on its host.
"""
import json
import hashlib
from pathlib import Path
import sys
import types
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from sq_reference import revision, load_module

HEARTLIB_REVISION = '5858ca8f1ffe58d62be7c007ea55f1033c5ca456'
ENCODER_REVISION = 'e6020510612d6653b73d6cd54393ef80929e3c81'
ENCODER_SHA256 = '19b3c0284fecac70983731433a213ad20eb08d096be3b9a287078171de0d4e79'
DECODER_REVISION = 'f889dab0532cfa4bf459f2a3367eb6d346b8eeda'

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def load_encoder(heartlib, encoder_snapshot, decoder_snapshot):
    import subprocess
    if revision(heartlib) != HEARTLIB_REVISION or subprocess.check_output(
            ['git', '-C', str(heartlib), 'status', '--porcelain'], text=True).strip():
        raise ValueError('Expected clean pinned HeartLib source')
    encoder_snapshot, decoder_snapshot = Path(encoder_snapshot), Path(decoder_snapshot)
    if encoder_snapshot.name != ENCODER_REVISION or decoder_snapshot.name != DECODER_REVISION:
        raise ValueError('Unexpected encoder/decoder checkpoint revision')
    if digest(encoder_snapshot / 'encoder.safetensors') != ENCODER_SHA256:
        raise ValueError('Encoder weights checksum mismatch')
    package = types.ModuleType('heartcodec_encoder_reference')
    package.__path__ = [str(Path(heartlib) / 'src/heartlib/heartcodec/models')]
    sys.modules[package.__name__] = package
    module = load_module(package.__name__ + '.encoder', Path(package.__path__[0]) / 'encoder.py')
    config = json.loads((encoder_snapshot / 'config.json').read_text())
    encoder = module.HeartCodecEncoder(config)
    state = load_file(str(encoder_snapshot / 'encoder.safetensors'))
    if not state or any(not name.startswith('encoder.') for name in state):
        raise ValueError('Unexpected encoder checkpoint names')
    encoder.load_state_dict({name[len('encoder.'):]: value for name, value in state.items()}, strict=True, assign=True)
    from vector_quantize_pytorch import ResidualVQ
    settings = json.loads((decoder_snapshot / 'config.json').read_text())
    keys = ('dim', 'codebook_size', 'decay', 'commitment_weight', 'threshold_ema_dead_code',
            'use_cosine_sim', 'codebook_dim', 'num_quantizers')
    quantizer = ResidualVQ(**{key: settings[key] for key in keys})
    index = json.loads((decoder_snapshot / 'model.safetensors.index.json').read_text())['weight_map']
    prefix = 'flow_matching.vq_embed.'
    rvq_state = {}
    for shard in sorted({file for name, file in index.items() if name.startswith(prefix)}):
        with safe_open(decoder_snapshot / shard, framework='pt', device='cpu') as source:
            for name in source.keys():
                if name.startswith(prefix):
                    rvq_state[name[len(prefix):]] = source.get_tensor(name)
    quantizer.load_state_dict(rvq_state, strict=True, assign=True)
    return encoder.eval().requires_grad_(False), quantizer.eval().requires_grad_(False), config

def runtime_name(name):
    for source, target in (
        ('feature_extractors.whisper_encoder', 'wh'),
        ('feature_extractors.wavlm_encoder', 'wl'),
        ('feature_extractors.muencoder', 'mu'),
        ('feature_extractors.resample_24k_to_16k', 'rs16'),
        ('.encoder_transformers', '.blocks'),
        ('.cond_fusion_layer_acoustic', '.acoustic'),
        ('.cond_fusion_layer_semantic', '.semantic'),
        ('.cond_fusion_layer_phone', '.phone'),
        ('.conformer', '.enc'), ('.encoder', '.enc'),
        ('.feature_extractor.conv_layers', '.conv'),
        ('.feature_projection', '.proj'), ('.pos_conv_embed', '.pos'),
        ('.intermediate_dense', '.up'), ('.output_dense', '.down'),
        ('.feed_forward', '.ffn'), ('.self_attn', '.attn'),
        ('.attention', '.attn'), ('.layer_norm', '.norm'),
        ('.conv_module', '.conv'),
        ('preprocessor_melspec_2048', 'mel'),
    ):
        name = name.replace(source, target)
    if len(name.encode()) >= 64:
        raise ValueError('GGML tensor name too long: ' + name)
    return name

def runtime_tensors(encoder, quantizer):
    """Materialize effective parametrized weights without mutating the oracle."""
    consumed = set()
    result = {}
    mapping = {}
    def add(name, value):
        key = runtime_name(name)
        if key in result:
            raise ValueError('Runtime tensor collision: ' + key)
        result[key] = value.detach().float().cpu().contiguous()
        mapping[name] = key
    for name, child in encoder.named_modules():
        if hasattr(child, 'parametrizations') and 'weight' in child.parametrizations:
            add(name + '.weight', child.weight)
            consumed.update(key for key in encoder.state_dict() if key.startswith(name + '.parametrizations.weight.'))
    for name, value in encoder.state_dict().items():
        if name not in consumed:
            if value.is_floating_point():
                add(name, value)
            elif not name.endswith('num_batches_tracked'):
                raise ValueError('Unexpected integer buffer: ' + name)
    add('rs16.kernel', encoder.feature_extractors.resample_24k_to_16k.kernel)
    add('wh.mel_filters', torch.as_tensor(encoder.feature_extractors.whisper_processor.mel_filters, dtype=torch.float32))
    add('wh.window', torch.hann_window(400))
    for name, value in quantizer.state_dict().items():
        if name.startswith('project_in.') or name.endswith('._codebook.embed'):
            add('rvq.' + name, value.squeeze(0) if name.endswith('._codebook.embed') else value)
    return result, mapping
