"""
run_inference.py — ONNX inference service for channel flow ML closure.

Reads feature batches from Redis, runs ChannelCorrectionNet ONNX model,
pushes delta_nu_t+ corrections back to the channel solver via RPUSH.
No conduit dependency — catalyst_ml uses numpy savez on the wire.
"""
import argparse, sys
import numpy as np
import os
sys.path.insert(0, "../../")
try:
    import onnxruntime as ort
except ImportError:
    raise ImportError("pip install onnxruntime")

from paraview.simple import *

from paraview.vtk.numpy_interface import dataset_adapter as dsa
from vtk.util import numpy_support
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkUnstructuredGrid
import vtkmodules.vtkCommonDataModel as vtk

CHANNEL = "ml_closure"
RE_TAU  = 180.0


xml_plugin_path = os.path.abspath("/home/local/KHQ/ayman.yousef/Downloads/paraview_new_fides_writer/paraview/Remoting/Application/Resources/proxies_fides.xml")
LoadPlugin(xml_plugin_path, ns=globals())


NotReady = 1
EndOfStream = 2

fides=FidesJSONReader()
fides.DataSourceEngines = ["source", "SST"]
fides.DataSourcePath = ["source", "/home/local/KHQ/ayman.yousef/Downloads/ayman_internship_work/ayman_internship_work/test_write.gp"]
fides.StreamSteps = 1
fides.FileName = "./flow.json"
fides.PrepareNextStep()
fides.UpdatePipelineInformation()

## ! Also generate a Writer. It writes data back to sim 
fides_write=FidesWriter()
fides_write.FileName = "./inference_service.bp"
fides_write.Engine = "BP"

# fides_write.Input # ! Need to generate a VTK object from inferred data 

def read_from_fides():
    global fides

    status = NotReady
    while status == NotReady:
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

    return u_plus, du_dy, nu_t, coords
    # Gather the actual arrays from here 

def fides_write_back(delta_nu_t):

    # * Generate vtk unstructure grid
    output_grid = vtk.vtkUnstructuredGrid()

    coords_x = np.arange(1.0, 181.0, dtype=np.float64)
    coords_y = np.zeros([180])
    coords_z = np.zeros([180])

    connectivity_flat = np.arange(0, 180, dtype=np.int64)
    # * Set it up as the input to the fides_writer

    #coords = np.concatenate((coords_x, coords_y, coords_z))
    coords = np.stack((coords_x, coords_y,coords_z), axis=-1)

    # 3. Set Points
    pts = vtkPoints()
    pts.SetData(numpy_support.numpy_to_vtk(coords, deep=True))
    output_grid.SetPoints(pts)

    # 4. Set Cells (Using VTK cell array format)
    cells = vtkCellArray()
    # Flatten and format for VTK: each cell starts with its number of points

    num_cells = 180

    vtk_conn_array = numpy_support.numpy_to_vtkIdTypeArray(np.array(connectivity_flat, dtype=np.int64), deep=True)
    cells.SetCells(num_cells, vtk_conn_array)

    # Assign cells (using VTK_TRIANGLE or generic cell types structure)
    # For mixed types, use SetCells with offsets and connectivity arrays in newer VTK versions
    output_grid.SetCells(vtk.VTK_LINE, cells) # or specify cell types array explicitly

    # 5. Attach NumPy Field Data to Points
    vtk_field = numpy_support.numpy_to_vtk(delta_nu_t, deep=True)
    vtk_field.SetName("delta_nu_t")
    output_grid.GetPointData().AddArray(vtk_field)
    output_grid.GetPointData().SetActiveScalars("delta_nu_t")


    tp = TrivialProducer()

    # 2. Inject your raw VTK unstructured grid into it
    tp.GetClientSideObject().SetOutput(output_grid)

    fides_write.Input = tp
    fides_write.UpdatePipeline()

def run(model_path, redis, channel):
    session   = ort.InferenceSession(model_path)
    in_name   = session.get_inputs()[0].name

    print(f"[inference] model={model_path} ", flush=True)

    steps = 1
    misses = 0
    for step in range(steps):

        print(f"[inference] step={step}", flush=True)

        u_plus, du_dy, nu_t, coords = read_from_fides()

        u_plus = u_plus.astype(np.float32)
        du_dy  = du_dy.astype(np.float32)
        nu_t   = nu_t.astype(np.float32)

        features = np.stack([
            u_plus / 20.0,
            np.abs(du_dy) * RE_TAU / 20.0,
            np.log1p(nu_t) / 4.0,
        ], axis=1)

        delta_nu_t = session.run(None, {in_name: features})[0].squeeze()

        print(f"[fides write back] step={step}", flush=True)

        fides_write_back(delta_nu_t)

        if step % 20 == 0:
            print(f"[inference] step={step}, delta_nu_t={delta_nu_t}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",   default="channel_correction.onnx")
    ap.add_argument("--redis",   default="localhost:6379")
    ap.add_argument("--channel", default=CHANNEL)
    args = ap.parse_args()
    run(args.model, args.redis, args.channel)
