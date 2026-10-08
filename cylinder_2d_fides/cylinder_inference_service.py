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

logging.basicConfig(filename="infer_fides.log",filemode='w', level=logging.INFO, format='[INFER] %(message)s')
logger = logging.getLogger(__name__)

from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridWriter

xml_plugin_path = os.path.abspath("/home/local/KHQ/ayman.yousef/Downloads/paraview_new_fides_writer/paraview/Remoting/Application/Resources/proxies_fides.xml")
LoadPlugin(xml_plugin_path, ns=globals())


NotReady = 1
EndOfStream = 2

fides=FidesJSONReader()
fides.DataSourceEngines = ["source", "SST"]
fides.DataSourcePath = ["source", "/home/local/KHQ/ayman.yousef/Downloads/catalyst-ml/catalyst-ml/examples/2d_cyl.gp"]
fides.StreamSteps = 1
fides.FileName = "./flow.json"
fides.PrepareNextStep()
fides.UpdatePipelineInformation()

## ! Also generate a Writer. It writes data back to sim 
fides_write=FidesWriter()
fides_write.FileName = "./inference_service.bp"
fides_write.Engine = "BP"

def read_from_fides(step):
    global fides

    t0 = time.monotonic()

    status = NotReady
    while status == NotReady:
        # must call PrepareNextStep to get Fides ready to read the
        # next step
        fides.PrepareNextStep()
        fides.UpdatePipelineInformation()
        status = fides.NextStepStatus
    dt = (time.monotonic() - t0)
    logger.info('Step {}, Waiting on Writer, dt={}'.format(step, dt))

    t0 = time.monotonic()
    fides.UpdatePipeline()

    merged_source = MergeBlocks(fides)

    data = paraview.servermanager.Fetch(merged_source)
    point_data = data.GetPointData()

    omega = numpy_support.vtk_to_numpy(point_data.GetArray("omega_coarse"))
    ux =  numpy_support.vtk_to_numpy(point_data.GetArray("ux"))
    uy =  numpy_support.vtk_to_numpy(point_data.GetArray("uy"))

    dt = (time.monotonic() - t0)
    logger.info('Step {}, Actual Read Op., dt={}'.format(step, dt))

    return omega.reshape(150,50), ux.reshape(150,50), uy.reshape(150,50)
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

def fides_write_back(omega_rom, ux, uy):

    # * Generate vtk unstructure grid
    nx = 150
    ny = 50
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
    #coords = np.stack((ux, uy,coords_z), axis=-1)

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

    # 5. Attach NumPy Field Data to Points
    vtk_field = numpy_support.numpy_to_vtk(omega_rom.flatten(), deep=True)
    vtk_field.SetName("omega_rom")

    output_grid.GetPointData().AddArray(vtk_field)
    output_grid.GetPointData().SetActiveScalars("omega_rom")

    tp = TrivialProducer()

    # 2. Inject your raw VTK unstructured grid into it
    tp.GetClientSideObject().SetOutput(output_grid)

    fides_write.Input = tp
    fides_write.UpdatePipeline()

    writer = vtkXMLUnstructuredGridWriter()
    writer.SetFileName("output_mesh.vtu")
    writer.SetInputData(output_grid)

    # Optional: set compression options
    writer.SetDataModeToBinary() # or SetDataModeToAscii()

    writer.Write()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/cylinder_fno.pt')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max-steps', type=int, default=50)
    args = parser.parse_args()

    
    model = CylinderFNO(ic=3, oc=3, w=32, nb=4, mx=12, my=8).to(args.device).eval()
    model.load_state_dict(torch.load(args.model, map_location=args.device))
    backend = TorchBackend(model, device=args.device)

    logger.info('Inference service running (backend=%s)', backend.name)
    step = 0

    while step < args.max_steps:
        #batch = transport.read_pending_batch(step, timeout=10.0)

        print(" ====================== STEP {} ======================".format(step))
        t0 = time.monotonic()

        omega, ux, uy = read_from_fides(step) # * Is timing affected by hangs? 
        dt = (time.monotonic() - t0)
        logger.info('Step {}, Read From Fides, dt={}'.format(step, dt))
        features = {
            'omega': omega,
            'ux': ux,
            'uy': uy
        }

        t0 = time.monotonic()

        x, ranges = preprocess(features)

        output = backend.infer(x)[0]  # (3, 150, 50)
        result = postprocess(output, ranges)

        dt = (time.monotonic() - t0)

        #logger.info('Step %d: omega coarse=[%.5f,%.5f] rom=[%.5f,%.5f] (%s, %.1fms)',
        #            step,
        #            features['omega'].min(), features['omega'].max(),
        #            result['omega_rom'].min(), result['omega_rom'].max(),
        #            backend.name, dt)
        logger.info('Step {}, Model Inference, dt={}'.format(step, dt))
        # Now write to BP to allow for transfer back to the simulator
        t0 = time.monotonic()
        fides_write_back(result['omega_rom'], ux, uy)
        dt = (time.monotonic() - t0)
        logger.info('Step {}, Fides Write Back, dt={}'.format(step, dt))
        step += 1

    logger.info('Done.')


if __name__ == '__main__':
    main()
