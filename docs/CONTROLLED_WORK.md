# Controlled codec work

`heartcodec_get_work_api(1)` from `heartcodec_work.h` negotiates cooperative
cancellation and progress for waveform encoding and token-to-stereo decoding.
Core, flow, encoder and RNG tables/layouts/defaults are unchanged. This follows
sa3.cpp's worker callback/owned-result pattern and ACE-Step's step cancellation.
It adds no GGML operations or dependency changes.

Initialize a size-prefixed control, set `should_cancel`, `on_progress` and `user`,
then call its `encode` or `tokens_decode` entry on a serialized codec context.
Callbacks execute synchronously on that worker and must not throw or reenter.
Use an atomic flag if another thread signals cancellation. Never run inference
or callbacks on the DAW audio thread. Null callbacks permit ordinary execution.

The new functions return 0/1/2 for success/invalid argument/runtime error and
`HEARTCODEC_WORK_CANCELLED_V1` (3) for observed cancellation. Failure/cancellation
leaves output unchanged; initialize an empty result first. Results use the
existing encoder/stereo free functions and survive context destruction. Contexts
remain usable after cancellation; controlled-call observers are detached on
every exit. Model loading or graph work completed before cancellation is not
rolled back, and diagnostic dump files, if requested, are not removed.

Polling occurs before/after backend graphs, between encoder chunks, during lazy
component tensor loading and between decoder flow steps/windows and SQ channels.
It cannot interrupt a running graph. Initial context creation/loading uses the
existing API and is not interruptible through this control. Progress reports
completed RVQ frames for encoding, completed flow windows and completed SQ
window/channels for decoding; it is coarser than cancellation polling. There is
no watchdog or wall-time cancellation guarantee. Staged decoding still releases
flow weights/graphs before loading SQ.

`test-work` exercises negotiation without models. Its optional integration form
is `test-work ENCODER.gguf FLOW.gguf SQ.gguf cpu|cuda`: it checks cancellation
before work, during encoder graphs, after an encoder chunk and after a flow
window; it then reuses the same contexts through legacy and controlled calls
and compares codes/waveforms byte-for-byte. This is an ownership/behavior
regression, not a new upstream numerical-parity claim.

`test-rng` also verifies bulk versus chunked normal/exponential draws with a full
64-bit seed, nonzero offset and target cap 288. MuLaCover can request eight
sampling heads per frame without retaining noise for the entire duration. The
existing Torch CUDA capability remains the single RNG implementation; CPU-only
builds still report generation unavailable with no fallback.

Actual build, integration and RNG checks are recorded in
[the controlled-work report](validation/controlled-work.json).
