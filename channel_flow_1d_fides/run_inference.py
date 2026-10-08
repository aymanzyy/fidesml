"""
run_inference.py — ONNX inference service for channel flow ML closure.

Reads feature batches from Redis, runs ChannelCorrectionNet ONNX model,
pushes delta_nu_t+ corrections back to the channel solver via RPUSH.
No conduit dependency — catalyst_ml uses numpy savez on the wire.
"""
import argparse, sys
import numpy as np

sys.path.insert(0, "../../")
from catalyst_ml.transport import RedisTransport

try:
    import onnxruntime as ort
except ImportError:
    raise ImportError("pip install onnxruntime")

CHANNEL = "ml_closure"
RE_TAU  = 180.0


def run(model_path, redis, channel):
    transport = RedisTransport(endpoint=redis, channel=channel,
                               max_pending_steps=0)
    session   = ort.InferenceSession(model_path)
    in_name   = session.get_inputs()[0].name

    print(f"[inference] model={model_path}  Redis={redis}", flush=True)

    step   = 0
    misses = 0
    while True:
        batch = transport.read_pending_batch(step=step, timeout=45.0)
        if not batch:
            misses += 1
            if misses >= 3:
                print(f"[inference] {misses} timeouts at step {step}, stopping", flush=True)
                break
            print(f"[inference] timeout at step {step}, retry {misses}/3", flush=True)
            continue
        misses = 0

        for rank, fields in batch.items():
            u_plus = fields["u_plus"].astype(np.float32)
            du_dy  = fields["du_dy"].astype(np.float32)
            nu_t   = fields["nu_t"].astype(np.float32)

            features = np.stack([
                u_plus / 20.0,
                np.abs(du_dy) * RE_TAU / 20.0,
                np.log1p(nu_t) / 4.0,
            ], axis=1)

            delta_nu_t = session.run(None, {in_name: features})[0].squeeze()
            transport.push_results(rank=rank, step=step,
                                   fields={"delta_nu_t": delta_nu_t})

        if step % 20 == 0:
            print(f"[inference] step={step}", flush=True)
        step += 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",   default="channel_correction.onnx")
    ap.add_argument("--redis",   default="localhost:6379")
    ap.add_argument("--channel", default=CHANNEL)
    args = ap.parse_args()
    run(args.model, args.redis, args.channel)
