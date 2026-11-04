# Third-party source and model inventory

This project contains inspection/reference tooling and native SQ, RVQ/flow and
waveform-encoder graphs based on the documented HeartCodec architecture and SA3 GGML graph
patterns. Upstream Python source checkouts are siblings outside the project. Model
weights and generated GGUFs are not redistributed here. The shared GGML runtime is
included as a pinned submodule with its own license.

| Reference | Observed source license / status |
| --- | --- |
| [HeartMuLa/heartlib](https://github.com/HeartMuLa/heartlib) | Apache-2.0 source; frontend contains MusicFM/ByteDance attribution and MIT permission text which must be preserved when ported |
| [mimbres/spansynth-edit](https://github.com/mimbres/spansynth-edit) | Apache-2.0 source; inspect codec attribution before copying implementation |
| [HeartMuLa/MuLaCover](https://github.com/HeartMuLa/MuLaCover) | Apache-2.0 source; separate MODEL_LICENSE restricts weights and outputs to noncommercial use |
| [betweentwomidnights/ggml](https://github.com/betweentwomidnights/ggml) | Required shared runtime fork; preserve its license when adding dependency |

## Waveform encoder attribution

The native encoder reimplements the inference path in HeartLib revision
5858ca8f1ffe58d62be7c007ea55f1033c5ca456, together with the Whisper, WavLM and
Wav2Vec2-Conformer behavior in Transformers 4.57.6. Those reference sources use
Apache-2.0; a copy is included in `third_party/LICENSE-Apache-2.0.txt`.
The MuEncoder frontend retains MusicFM / ByteDance's MIT copyright and permission
text in `third_party/LICENSE-MusicFM.txt`.

The residual-codebook distance convention follows vector-quantize-pytorch 1.27.15;
the new native implementation uses the original decoder's trained codebooks and
does not bundle its Python package. Native graph, preprocessing, ABI and CLI work
is maintained here. Reference Python/third-party checkouts and all checkpoint
payloads remain outside this repository's distributed source.

Before distributing converted weights, record the exact checkpoint license at the
pinned revision for HeartCodec decoder, HeartCodec encoder, Qwen, and every embedded
feature model. These checkpoint terms have not yet been fully audited. Repository
source licensing alone does not establish model-weight redistribution rights.

## RNG source attribution

The Philox integer rounds in `src/heartcodec-rng.cu` adapt the sibling
sa3.cpp `src/rng.h` implementation at 95508283b303164aa6a0b5a5140dfe79d4c489b3.
Distribution layout/counter advancement and exponential clamp semantics follow
PyTorch ATen CUDA distribution kernels (the pinned oracle uses Torch 2.7.1).
CUDA math intrinsics preserve the cuRAND-style transforms; no cuRAND header
or library is bundled. ACE-Step's `src/philox.h` was consulted, not copied.

### sa3.cpp (MIT)

MIT License

Copyright (c) 2026 betweentwomidnights

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

### PyTorch (BSD-3-Clause)

From PyTorch:

Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU                      (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

From Caffe2:

Copyright (c) 2016-present, Facebook Inc. All rights reserved.

All contributions by Facebook:
Copyright (c) 2016 Facebook Inc.

All contributions by Google:
Copyright (c) 2015 Google Inc.
All rights reserved.

All contributions by Yangqing Jia:
Copyright (c) 2015 Yangqing Jia
All rights reserved.

All contributions by Kakao Brain:
Copyright 2019-2020 Kakao Brain

All contributions by Cruise LLC:
Copyright (c) 2022 Cruise LLC.
All rights reserved.

All contributions by Tri Dao:
Copyright (c) 2024 Tri Dao.
All rights reserved.

All contributions by Arm:
Copyright (c) 2021, 2023-2024 Arm Limited and/or its affiliates

All contributions from Caffe:
Copyright(c) 2013, 2014, 2015, the respective contributors
All rights reserved.

All other contributions:
Copyright(c) 2015, 2016 the respective contributors
All rights reserved.

Caffe2 uses a copyright model similar to Caffe: each contributor holds
copyright over their contributions to Caffe2. The project versioning records
all such contribution and copyright details. If a contributor wants to further
mark their specific copyright on a particular contribution, they should
indicate their copyright solely in the commit message of the change when it is
committed.

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
   and IDIAP Research Institute nor the names of its contributors may be
   used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
