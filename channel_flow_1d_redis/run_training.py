"""
run_training.py — training entry point for channel flow ML closure.

Reads feature batches from Redis, looks up reference delta_nu_t+ via
SpatialReferenceData, trains ChannelCorrectionNet (PyTorch MLP), exports ONNX.
"""
import argparse
import sys
import os
import numpy as np
import time
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', '..'))
sys.path.insert(0, _here)

import torch
from catalyst_ml.transport import RedisTransport
from catalyst_ml import SpatialReferenceData
from model import make_model, channel_loss, export_onnx

RE_TAU  = 180.0
CHANNEL = "ml_closure"


def make_features(u_plus, du_dy, nu_t):
    f0 = (u_plus / 20.0).astype(np.float32)
    f1 = (np.abs(du_dy) * RE_TAU / 20.0).astype(np.float32)
    f2 = (np.log1p(nu_t) / 4.0).astype(np.float32)
    return np.stack([f0, f1, f2], axis=1)


def run(ref_path, redis, channel, steps, lr, export):
    transport = RedisTransport(endpoint=redis, channel=channel, max_pending_steps=0)
    ref_data  = SpatialReferenceData(ref_path, field='correction')
    model     = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"[training] waiting for features  ref={ref_path}  steps={steps}")

    reads_from_redis = []
    train_loops = []

    for step in range(steps):
        before_read = time.monotonic()
        batch = transport.read_pending_batch(step=step, timeout=30.0)
        read_time =  time.monotonic() - before_read
        reads_from_redis.append(read_time)

        before_train = time.monotonic()

        if not batch:
            print(f"[training] step {step}: timed out, stopping")
            break
        for rank, fields in batch.items():
            u_plus = np.array(fields['u_plus'], dtype=np.float64)
            du_dy  = np.array(fields['du_dy'],  dtype=np.float64)
            nu_t   = np.array(fields['nu_t'],   dtype=np.float64)
            coords = np.array(fields['coords_2d'], dtype=np.float64).reshape(-1, 2)
    
            X          = make_features(u_plus, du_dy, nu_t)
            target, valid = ref_data.lookup(coords)
            target     = target.astype(np.float32).squeeze()
            X[~valid]  = 0.0
            target[~valid] = 0.0

            pred = model(torch.tensor(X))
            loss = channel_loss(pred, torch.tensor(target))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_time =  time.monotonic() - before_train
        train_loops.append(train_time)

        if step % 50 == 0:
            print(f"[training] step={step:4d}  loss={loss.item():.6f}")
    export_onnx(model, export)

    avg_regis_read = np.mean(np.array(reads_from_redis))
    avg_train_time = np.mean(np.array(train_loops))

    print(" ++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("Avg Training Time: {} secs".format(avg_train_time))
    print("Avg Pull From Redis Time: {} secs".format(avg_regis_read))
    print(" ++++++++++++++++++++++++++++++++++++++++++++++++++++++")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref",    default="channel_ref.npz")
    ap.add_argument("--redis",  default="localhost:6379")
    ap.add_argument("--channel",default=CHANNEL)
    ap.add_argument("--steps",  type=int,   default=400)
    ap.add_argument("--lr",     type=float, default=1e-3)
    ap.add_argument("--export", default="channel_correction.onnx")
    args = ap.parse_args()
    run(args.ref, args.redis, args.channel, args.steps, args.lr, args.export)
