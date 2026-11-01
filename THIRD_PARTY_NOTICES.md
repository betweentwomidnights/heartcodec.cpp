# Third-party source and model inventory

This project contains inspection/reference tooling and a native Base SQ decoder
graph based on the documented HeartCodec architecture and the SA3 GGML graph
patterns. Upstream Python source checkouts are siblings outside the project. Model
weights and generated GGUFs are not redistributed here. The shared GGML runtime is
included as a pinned submodule with its own license.

| Reference | Observed source license / status |
| --- | --- |
| [HeartMuLa/heartlib](https://github.com/HeartMuLa/heartlib) | Apache-2.0 source; frontend contains MusicFM/ByteDance attribution and MIT permission text which must be preserved when ported |
| [mimbres/spansynth-edit](https://github.com/mimbres/spansynth-edit) | Apache-2.0 source; inspect codec attribution before copying implementation |
| [HeartMuLa/MuLaCover](https://github.com/HeartMuLa/MuLaCover) | Apache-2.0 source; separate MODEL_LICENSE restricts weights and outputs to noncommercial use |
| [betweentwomidnights/ggml](https://github.com/betweentwomidnights/ggml) | Required shared runtime fork; preserve its license when adding dependency |

Before distributing converted weights, record the exact checkpoint license at the
pinned revision for HeartCodec decoder, HeartCodec encoder, Qwen, and every embedded
feature model. These checkpoint terms have not yet been fully audited. Repository
source licensing alone does not establish model-weight redistribution rights.
