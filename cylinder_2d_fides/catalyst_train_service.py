#!/usr/bin/env python3
"""
cylinder_inference_service.py - FNO inference for cylinder flow via catalyst-ml.

Uses per-sample normalization: normalize input by its own 1st/99th percentile,
run FNO, denormalize output by the same range (since both were trained with
combined ranges, the FNO output is in the same normalized space as the input).
"""
import sys

import numpy as np
import torch
import time
import logging
import os
from fno2d_cylinder import CylinderFNO
from catalyst_ml.transport import RedisTransport
from catalyst_ml.inference import TorchBackend

from paraview.simple import *

from paraview.vtk.numpy_interface import dataset_adapter as dsa

from vtk.util import numpy_support
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkUnstructuredGrid
import vtkmodules.vtkCommonDataModel as vtk
import math
sys.path.append("/home/local/KHQ/ayman.yousef/Downloads/catalyst-ml/catalyst-ml/examples/cylinder_2d_fides")
from pn_autoencoder import PointCloudAE
import torch.nn as nn
logging.basicConfig(filename="train_fides.log",filemode='w',level=logging.INFO, format='[INFER] %(message)s')
logger = logging.getLogger(__name__)
from paraview import catalyst
from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridWriter
xml_plugin_path = os.path.abspath("/home/local/KHQ/ayman.yousef/Downloads/paraview_new_fides_writer/paraview/Remoting/Application/Resources/proxies_fides.xml")
LoadPlugin(xml_plugin_path, ns=globals())


NotReady = 1
EndOfStream = 2

NUMPOINTS=7500
LATENTSIZE=128

fides=FidesJSONReader()
fides.DataSourceEngines = ["source", "SST"]
fides.DataSourcePath = ["source", "/home/local/KHQ/ayman.yousef/Downloads/catalyst-ml/catalyst-ml/examples/2d_cyl.gp"]
fides.StreamSteps = 1
fides.FileName = "./flow.json"
fides.PrepareNextStep()
fides.UpdatePipelineInformation()

## * Generate writer to pass back loss  
fides_write=FidesWriter()
fides_write.FileName = "./training_service.bp"
fides_write.Engine = "BP"

model = PointCloudAE(NUMPOINTS, LATENTSIZE)
#backend = TorchBackend(model, device=args.device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
global_epoch = 0
training_losses = []

def make_view():
    rv = CreateView('RenderView')
    rv.ViewSize = [1200, 400]
    rv.Background = [1.0, 1.0, 1.0]
    rv.OrientationAxesVisibility = 0
    rv.InteractionMode = '2D'
    rv.CameraPosition = [75, 25, 100]
    rv.CameraFocalPoint = [75, 25, 0]
    rv.CameraViewUp = [0, 1, 0]
    rv.CameraParallelProjection = 1
    rv.CameraParallelScale = 30
    return rv

def fides_write_back(ux, uy, training_loss):

    nx = 150
    ny = 50

    # * Set training_loss as POINT data
    # * Copy the value per point to write back correctly

    training_loss_arr = np.zeros((nx * ny))
    training_loss_arr.fill(training_loss)

    # * Generate vtk unstructure grid

    output_grid = vtk.vtkUnstructuredGrid()
    _x = np.linspace(0, nx - 1, nx).astype("float64")
    _y = np.linspace(0, ny - 1, ny).astype("float64")
    coords_z = np.zeros(nx * ny)

    coords_x, coords_y = np.meshgrid(_x, _y, indexing='ij')

    num_cells = (nx-1) * (ny-1) 

    i_idx, j_idx = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), indexing='ij')
    p0 = i_idx * ny + j_idx
    p1 = (i_idx + 1) * ny + j_idx
    p2 = (i_idx + 1) * ny + (j_idx + 1)
    p3 = i_idx * ny + (j_idx + 1)

    offsets = np.arange(0, (num_cells + 1) * 4, 4, dtype=np.int64)
    conn = np.stack([p0, p1, p2, p3], axis=-1).flatten()
    connectivity = conn.flatten().astype(np.int64)

    coords = np.stack((coords_x.flatten(), coords_y.flatten(), coords_z), axis=-1)

    pts = vtkPoints()
    pts.SetData(numpy_support.numpy_to_vtk(coords, deep=True))
    output_grid.SetPoints(pts)

    cells = vtkCellArray()

    vtk_offsets = numpy_support.numpy_to_vtkIdTypeArray(offsets, deep=True)
    vtk_conn_array = numpy_support.numpy_to_vtkIdTypeArray(connectivity, deep=True)

    cells.SetData(vtk_offsets, vtk_conn_array)

    cell_types = np.full(num_cells, vtk.VTK_QUAD, dtype=np.uint8)
    vtk_cell_types = numpy_support.numpy_to_vtk(cell_types, deep=True)

    output_grid.SetCells(vtk_cell_types, cells) # or specify cell types array explicitly

    loss_vtk_field = numpy_support.numpy_to_vtk(training_loss_arr, deep=True)
    loss_vtk_field.SetName("train_loss")

    output_grid.GetPointData().AddArray(loss_vtk_field)
    output_grid.GetPointData().SetActiveScalars("train_loss")

    tp = TrivialProducer()

    tp.GetClientSideObject().SetOutput(output_grid)

    fides_write.Input = tp
    fides_write.UpdatePipeline()

def visualize(model_pred, step):

    global fides

    # Omega coarse
    rv1 = make_view()
    d1 = Show(fides, rv1)
    d1.Representation = 'Surface'
    ColorBy(d1, ('POINTS', 'omega_coarse'))
    lut1 = GetColorTransferFunction('omega_coarse')
    lut1.ApplyPreset('Cool to Warm', True)
    lut1.RescaleTransferFunction(-0.03, 0.03)
    d1.SetScalarBarVisibility(rv1, True)
    GetScalarBar(lut1, rv1).Title = 'omega (coarse)'

    output_gt_png = f"./catalyst_output/gt_output-{step:05d}.png"
    SaveScreenshot(output_gt_png, rv1, ImageResolution=[1200,400])

    # Predicted Omega + Coords

    ### * Generate grid from prediction
    t0 = time.monotonic()
    pred_tp = gen_tp_from_model(model_pred)
    dt = time.monotonic() - t0

    logger.info('Step {}, Generate TrivialProduce Call, dt={}'.format(step, dt))

    rv2 = make_view()
    cell_to_point = CellDatatoPointData(registrationName='CellToPoint', Input=pred_tp)

    d2 = Show(cell_to_point, rv2)
    d2.Representation = 'Surface'
    ColorBy(d2, ('POINTS', 'omega_coarse'))
    lut2 = GetColorTransferFunction('omega_coarse')
    lut2.ApplyPreset('Cool to Warm', True)
    lut2.RescaleTransferFunction(-0.03, 0.03)
    d2.SetScalarBarVisibility(rv2, True)
    GetScalarBar(lut2, rv2).Title = 'omega (coarse)'
    SetActiveView(rv2)

    output_pred_png = f"./catalyst_output/pred_output-{step:05d}.png"
    SaveScreenshot(output_pred_png, rv2, ImageResolution=[1200,400])


def gen_tp_from_model(prediction):

    output_grid = vtk.vtkUnstructuredGrid()

    coords_x = prediction[:, 0]
    coords_y = prediction[:, 1]
    coords_z = prediction[:, 2]

    omega_pred = prediction[:, 3] 
    vel_mag_pred = prediction[:, 4] 

    nx = 150
    ny = 50

    i_idx, j_idx = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), indexing='ij')
    p0 = i_idx * ny + j_idx
    p1 = (i_idx + 1) * ny + j_idx
    p2 = (i_idx + 1) * ny + (j_idx + 1)
    p3 = i_idx * ny + (j_idx + 1)

    conn = np.stack([p0, p1, p2, p3], axis=-1).flatten()
    num_cells = (nx - 1) * (ny - 1)

    coords = np.concatenate((np.expand_dims(coords_x, axis=-1), np.expand_dims(coords_y, axis=-1), np.expand_dims(coords_z, axis=-1)), axis=-1)
    offsets = np.arange(0, (num_cells + 1) * 4, 4, dtype=np.int64)
    connectivity = conn.flatten().astype(np.int64)

    vtk_offsets = numpy_support.numpy_to_vtkIdTypeArray(offsets, deep=True)
    vtk_connectivity = numpy_support.numpy_to_vtkIdTypeArray(connectivity, deep=True)

    pts = vtkPoints()
    pts.SetData(numpy_support.numpy_to_vtk(coords, deep=True))
    output_grid.SetPoints(pts)

    cells = vtkCellArray()
    cells.SetData(vtk_offsets, vtk_connectivity)

    cell_types = np.full(num_cells, vtk.VTK_QUAD, dtype=np.uint8)
    vtk_cell_types = numpy_support.numpy_to_vtk(cell_types, deep=True)
    output_grid.SetCells(vtk_cell_types, cells)


    vtk_field_omega = numpy_support.numpy_to_vtk(omega_pred.flatten(), deep=True)
    vtk_field_omega.SetName("omega_coarse")

    vtk_field_vel = numpy_support.numpy_to_vtk(vel_mag_pred.flatten(), deep=True)
    vtk_field_vel.SetName("vel_mag")

    output_grid.GetPointData().AddArray(vtk_field_omega)
    output_grid.GetPointData().SetActiveScalars("omega_coarse")

    output_grid.GetPointData().AddArray(vtk_field_vel)
    output_grid.GetPointData().SetActiveScalars("vel_mag")

    tp = TrivialProducer()
    tp.GetClientSideObject().SetOutput(output_grid)


    writer = vtkXMLUnstructuredGridWriter()
    writer.SetFileName("output_mesh.vtu")
    writer.SetInputData(output_grid)

    # Optional: set compression options
    writer.SetDataModeToBinary() # or SetDataModeToAscii()

    writer.Write()
    return tp

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

    omega = numpy_support.vtk_to_numpy(point_data.GetArray("omega_coarse"))
    ux =  numpy_support.vtk_to_numpy(point_data.GetArray("ux"))
    uy =  numpy_support.vtk_to_numpy(point_data.GetArray("uy"))

    coords = data.GetPoints().GetData()

    return omega, ux, uy, coords
    # Gather the actual arrays from here 


def normalize_robust(x, lo_pct=1, hi_pct=99, eps=1e-8):
    lo = np.percentile(x, lo_pct)
    hi = np.percentile(x, hi_pct)
    if hi - lo < eps:
        lo, hi = x.min() - eps, x.max() + eps
    return 2.0 * (np.clip(x, lo, hi) - lo) / (hi - lo) - 1.0, lo, hi


def denormalize(y, lo, hi):
    return (y + 1.0) / 2.0 * (hi - lo) + lo


def preprocess(features):
    """Normalize coarse state using its own per-channel percentiles."""
    channels = []
    ranges = []
    for name in ['omega', 'ux', 'uy']:
        normed, lo, hi = normalize_robust(features[name])
        channels.append(normed)
        ranges.append((lo, hi))
    st = np.stack(channels, axis=0)
    return st[np.newaxis, ...], ranges  # (1, 3, 150, 50), [(lo,hi)*3]


def postprocess(output, ranges):
    """Denormalize FNO output using the same ranges as input."""
    return {
        'omega_rom': denormalize(output[0], *ranges[0]).astype(np.float32),
        'ux_rom': denormalize(output[1], *ranges[1]).astype(np.float32),
        'uy_rom': denormalize(output[2], *ranges[2]).astype(np.float32),
    }


def train_loop(features, step):
    num_epochs = 5
    step_avg_loss = 0
    t0 = time.monotonic()

    global global_epoch, training_losses, model, optimizer, criterion
    for epoch in range(num_epochs):
        batch_temp = 0
        running_loss = 0.0
        optimizer.zero_grad()  
        outputs = model(features)
        loss = criterion(outputs, features)
        running_loss += loss.item()
        loss.backward()        
        optimizer.step() 
        batch_temp+=1
        epoch_loss = running_loss / len(features)
        training_losses.append(epoch_loss)
        step_avg_loss +=epoch_loss
    step_avg_loss = step_avg_loss/num_epochs
    global_epoch+=num_epochs

    dt = (time.monotonic() - t0)

    print("STEP {} AVG. LOSS: {}".format(step, step_avg_loss))

    # visualize the last prediction from the model
    logger.info('Step {}, Act. Training Loop, dt={}'.format(step, dt))

    t0 = time.monotonic()

    visualize(np.squeeze(outputs.detach().numpy(), axis=0), step)

    dt = (time.monotonic() - t0)

    logger.info('Step {}, Visualization of Pred, dt={}'.format(step, dt))
    return step_avg_loss

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/cylinder_fno.pt')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max-steps', type=int, default=50)
    args = parser.parse_args()

    step = 0

    while step < args.max_steps:

        print(" ====================== STEP {} ======================".format(step))
        t0 = time.monotonic()

        omega, ux, uy, coords = read_from_fides()
        dt = (time.monotonic() - t0)
        logger.info('Step {}, Read Data from Fides, dt={}'.format(step, dt))

        features = {
            'omega': omega,
            'ux': ux,
            'uy': uy
        }

        t0 = time.monotonic()
        vel_mag = np.sqrt(ux ** 2 + uy ** 2)

        omega = np.expand_dims(omega, axis=-1)
        vel_mag = np.expand_dims(vel_mag, axis =-1)

        features = np.concatenate((coords, omega , vel_mag), axis=-1)

        # * Run training

        x_tensor = torch.from_numpy(features).float()
        training_loss = train_loop(x_tensor, step)

        dt = (time.monotonic() - t0)

        logger.info('Step {}, Total Training Call, dt={}'.format(step, dt))


        # * Write Back Protocol

        fides_write_back(ux, uy, training_loss)


        step += 1

    logger.info('Done.')


if __name__ == '__main__':
    main()
