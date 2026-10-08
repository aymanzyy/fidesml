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
from paraview.simple import *

from paraview.vtk.numpy_interface import dataset_adapter as dsa

from vtk.util import numpy_support
RE_TAU  = 180.0
CHANNEL = "ml_closure"

xml_plugin_path = os.path.abspath("/home/local/KHQ/ayman.yousef/Downloads/paraview_new_fides_writer/paraview/Remoting/Application/Resources/proxies_fides.xml")
LoadPlugin(xml_plugin_path, ns=globals())

#sys.path.append("/home/local/KHQ/ayman.yousef/pytorch-env/venv/lib/python3.10/site-packages")

NotReady = 1
EndOfStream = 2
fides=FidesJSONReader()
fides.DataSourceEngines = ["source", "SST"]
fides.DataSourcePath = ["source", "/home/local/KHQ/ayman.yousef/Downloads/ayman_internship_work/ayman_internship_work/test_write.gp"]
fides.StreamSteps = 1
fides.FileName = "./flow.json"
fides.PrepareNextStep()
fides.UpdatePipelineInformation()

def read_from_fides():
    global fides

    status = NotReady
    while status == NotReady:
        # must call PrepareNextStep to get Fides ready to read the
        # next step
        fides.PrepareNextStep()
        fides.UpdatePipelineInformation()
        status = fides.NextStepStatus
    fides.UpdatePipeline()

    # fides object treated as PartitionedDataSetCollection 

    # surf_data = surface_producer.GetClientSideObject().GetOutputDataObject(0)

    merged_source = MergeBlocks(fides)

    data = paraview.servermanager.Fetch(merged_source)
    point_data = data.GetPointData()

    du_dy = numpy_support.vtk_to_numpy(point_data.GetArray("du_dy"))
    nu_t =  numpy_support.vtk_to_numpy(point_data.GetArray("nu_t"))
    u_plus =  numpy_support.vtk_to_numpy(point_data.GetArray("u_plus"))
    y_plus =  numpy_support.vtk_to_numpy(point_data.GetArray("y_plus"))

    coords = data.GetPoints().GetData()

    #.GetClientSideObject().GetOutputDataObject(0)

    # ! Just here for now

    return du_dy, nu_t, u_plus, y_plus, coords
    # Gather the actual arrays from here 
def make_features(u_plus, du_dy, nu_t):
    f0 = (u_plus / 20.0).astype(np.float32)
    f1 = (np.abs(du_dy) * RE_TAU / 20.0).astype(np.float32)
    f2 = (np.log1p(nu_t) / 4.0).astype(np.float32)
    return np.stack([f0, f1, f2], axis=1)


def run(ref_path, redis, channel, steps, lr, export):
    ref_data  = SpatialReferenceData(ref_path, field='correction')
    model     = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"[training] waiting for features  ref={ref_path}  steps={steps}")

    reads_from_fides = []
    train_loops = []

    for step in range(steps):
        ## Assuming a single batch for now

        #batch = read_from_fides();
        before_read = time.monotonic()

        du_dy, nu_t, u_plus,y_plus, coords = read_from_fides()

        read_time =  time.monotonic() - before_read
        reads_from_fides.append(read_time)

        before_train = time.monotonic()

        coords = coords[:, :2]
        coords = np.array(coords, dtype=np.float64).reshape(-1, 2)

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

    avg_fides_read = np.mean(np.array(reads_from_fides))
    avg_train_time = np.mean(np.array(train_loops))

    print(" ++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("Avg Training Time: {} secs".format(avg_train_time))
    print("Avg Pull From Fides Time: {} secs".format(avg_fides_read))
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
