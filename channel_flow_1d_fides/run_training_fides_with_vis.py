"""
run_training.py — training entry point for channel flow ML closure.

Reads feature batches from Redis, looks up reference delta_nu_t+ via
SpatialReferenceData, trains ChannelCorrectionNet (PyTorch MLP), exports ONNX.
"""
import argparse
import sys
import os
import numpy as np

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
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkUnstructuredGrid
import vtkmodules.vtkCommonDataModel as vtk
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


def SetupRenderView():
    # Create a new 'Render View'
    view = CreateView("RenderView")

    camera = GetActiveCamera()
    #camera.Azimuth(230)
    #camera.Elevation(-45)

    SetActiveView(None)

    view.ViewSize = [1600,800]
    view.CameraPosition = [0, 0, 0]
    # Disable palette override
    view.UseColorPaletteForBackground = 0

    # Set background color (example: pure White [1.0, 1.0, 1.0])
    view.Background = [0.0, 0.0, 0.0]
    #view.CameraFocalPoint = [19.452526958533134, 28.491610229010647, 10.883993417012459]
    #view.CameraViewUp = [0.07934883419275315, 0.953396338566962, -0.2910999555468221]
    #view.CameraFocalDisk = 1.0
    #view.CameraParallelScale = 54.99504523136608
    # create new layout object 'Layout #1'
    layout1 = CreateLayout(name="Layout #1")
    layout1.AssignView(0, view)
    #layout1.SetSize(824, 656)


    # restore active view
    SetActiveView(view)

    return view

def vis_from_new_prod(view, prod_):
    
    smooth_points = CellDatatoPointData(Input=prod_)

    #pipeline.UpdatePipeline()
    gridDisplay = Show(smooth_points, view, 'UnstructuredGridRepresentation')

    ColorBy(gridDisplay, ('POINTS', 'nu_t'))

    nu_tLUT = GetColorTransferFunction('nu_t')
    nu_tLUT.RescaleTransferFunction(0.0 , 640.0)


    nu_tPWF = GetOpacityTransferFunction('nu_t')
    nu_tPWF.RescaleTransferFunction(0.0, 640.0)

    gridDisplay.LookupTable = nu_tLUT


    # get color legend/bar for nu_tLUT in view view
    nu_tLUTColorBar = GetScalarBar(nu_tLUT, view)
    nu_tLUTColorBar.Title = 'nu_t'
    nu_tLUTColorBar.ComponentTitle = 'Magnitude'
    nu_tLUTColorBar.UseCustomLabels = 0 
    
    # show color legend
    gridDisplay.SetScalarBarVisibility(view, True)
    nu_tLUT.AutomaticRescaleRangeMode = 'Never' 

    SetActiveView(view)

    # create extractor
    if not os.path.isdir("./catalyst_renders"):
        os.mkdir("./catalyst_renders")

    output = f"./catalyst_renders/output-{cnt:05d}.png"
    SaveScreenshot(output, view, ImageResolution=[800, 800])


def dummy_vis(view, step):
    global fides
    gridDisplay = Show(fides, view, 'UnstructuredGridRepresentation')

    SetActiveView(view)
    #ResetCamera(view)
    # create extractor
    if not os.path.isdir("./catalyst_renders"):
        os.mkdir("./catalyst_renders")

    output = f"./catalyst_renders/output-{step:05d}.png"
    SaveScreenshot(output, view, ImageResolution=[800, 800])
def vis_from_fides(view, step):
    global fides
    smooth_points = CellDatatoPointData(Input=fides)

    #pipeline.UpdatePipeline()
    gridDisplay = Show(smooth_points, view, 'UnstructuredGridRepresentation')

    ColorBy(gridDisplay, ('POINTS', 'nu_t'))

    #gridDisplay.Representation = 'Surface'

    gridDisplay.Representation = 'Points'
    gridDisplay.PointSize = 8.0

    nu_tLUT = GetColorTransferFunction('nu_t')
    nu_tPWF = GetOpacityTransferFunction('nu_t')

    gridDisplay.LookupTable = nu_tLUT


    # get color legend/bar for nu_tLUT in view view
    nu_tLUTColorBar = GetScalarBar(nu_tLUT, view)
    nu_tLUTColorBar.Title = 'nu_t'
    nu_tLUTColorBar.ComponentTitle = 'Magnitude'
    nu_tLUTColorBar.UseCustomLabels = 0 
    
    # show color legend
    gridDisplay.SetScalarBarVisibility(view, True)
    nu_tLUT.AutomaticRescaleRangeMode = 'Never' 

    SetActiveView(view)
    #ResetCamera(view)
    # create extractor
    if not os.path.isdir("./catalyst_renders"):
        os.mkdir("./catalyst_renders")

    output = f"./catalyst_renders/output-{step:05d}.png"
    SaveScreenshot(output, view, ImageResolution=[800, 800])


def convert_back_to_tp():
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
    #vtk_field = numpy_support.numpy_to_vtk(delta_nu_t, deep=True)
    #vtk_field.SetName("delta_nu_t")
    #output_grid.GetPointData().AddArray(vtk_field)
    #output_grid.GetPointData().SetActiveScalars("delta_nu_t")


    tp = TrivialProducer()

    # 2. Inject your raw VTK unstructured grid into it
    tp.GetClientSideObject().SetOutput(output_grid)
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

    du_dy = numpy_support.vtk_to_numpy(point_data.GetArray("du_dy"))
    nu_t =  numpy_support.vtk_to_numpy(point_data.GetArray("nu_t"))
    u_plus =  numpy_support.vtk_to_numpy(point_data.GetArray("u_plus"))
    y_plus =  numpy_support.vtk_to_numpy(point_data.GetArray("y_plus"))

    coords = data.GetPoints().GetData()

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
    
    view = SetupRenderView()

    for step in range(steps):
        ## Assuming a single batch for now

        #batch = read_from_fides();
        
        du_dy, nu_t, u_plus,y_plus, coords = read_from_fides()

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

        ## Visualize every 50 steps as well, sync up with loss output
        if step % 50 == 0:
            print(f"[training] step={step:4d}  loss={loss.item():.6f}")
            # Have to convert, get a producer with delta nu t
            #new_prod = convert_back_to_tp()
            dummy_vis(view, step)
            #vis_from_fides(view, step)
            #vis_from_fides(view, new_prod)
    export_onnx(model, export)


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
