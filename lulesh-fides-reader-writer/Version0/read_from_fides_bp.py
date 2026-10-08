import argparse
import sys
import os
import numpy as np

# * Torch imports

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import torch.distributed as dist
from torch.multiprocessing import Process
from torch.utils.data import random_split
from torch.utils.data import Subset
import torch.optim as optim
from paraview.simple import *

#from pn_autoencoder import PointCloudAE

from paraview.vtk.numpy_interface import dataset_adapter as dsa

from vtk.util import numpy_support
from mpi4py import MPI
xml_plugin_path = os.path.abspath("/home/local/KHQ/ayman.yousef/Downloads/paraview_new_fides_writer/paraview/Remoting/Application/Resources/proxies_fides.xml")
LoadPlugin(xml_plugin_path, ns=globals())

## * Instantiate the model 

## * For multi-rank training

NUMPOINTS = 88795 #! Need to fix this so that Rank 1
LATENTSIZE = 4

myRank_ = int(MPI.COMM_WORLD.Get_rank())
worldSize_ = int(MPI.COMM_WORLD.Get_size())

#dist.init_process_group(backend="nccl", world_size=worldSize_, rank=myRank_)
#device_count = torch.cuda.device_count()

#os.environ['MASTER_ADDR'] = 'localhost'
#os.environ['MASTER_PORT'] = '12355'
#print("device_count is: {}".format(device_count))

##dist.init_process_group(backend="mpi")
#print("set process group")
#local_rank = myRank_ % device_count
#torch.cuda.set_device(0)

#print("Set Device")

# ! NO idea either of these for now
#model = PointCloudAE(NUMPOINTS, LATENTSIZE)
#device = torch.device("cuda:{}".format(0))
#model = model.to(device)
#ddp_model = torch.nn.parallel.DistributedDataParallel(model) 


print("Set DDP model")

#backend = TorchBackend(model, device=args.device)
#criterion = nn.MSELoss()
#optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#optimizer = optim.Adam(ddp_model.parameters(), lr=0.001)
#global_epoch = 0
#training_losses = []


#flow_json_str_name = './flow_rank{}.json'.format(str(myRank_)) 

flow_json_str_name = "/home/local/KHQ/ayman.yousef/Downloads/ayman_internship_work/ayman_internship_work/reader-lulesh-in-transit/Version0/first.json"

print("IM READING THIS SCRIPT: " + str(flow_json_str_name))

NotReady = 1
EndOfStream = 2
fides=FidesJSONReader()
fides.DataSourceEngines = ["source", "BP"]
fides.DataSourcePath = ["source", "./test_write.bp"]
fides.StreamSteps = 1
fides.FileName = flow_json_str_name
fides.PrepareNextStep()
fides.UpdatePipelineInformation()

print("THE READ WORKS") 

def train_loop(features, step):
    num_epochs = 5
    step_avg_loss = 0
    t0 = time.monotonic()

    global global_epoch, training_losses, model, optimizer, criterion
    for epoch in range(num_epochs):
        batch_temp = 0
        running_loss = 0.0
        optimizer.zero_grad()  
        outputs = ddp_model(features)
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

def SetupRenderView():
    # Create a new 'Render View'
    view = CreateView("RenderView")

    camera = GetActiveCamera()
    #camera.Azimuth(230)
    #camera.Elevation(-45)

    SetActiveView(None)

    view.ViewSize = [1600,800]
    view.CameraPosition = [157.90070691620653, 64.91180236667495, 167.90421495515105]
    view.CameraFocalPoint = [19.452526958533134, 28.491610229010647, 10.883993417012459]
    view.CameraViewUp = [0.07934883419275315, 0.953396338566962, -0.2910999555468221]
    view.CameraFocalDisk = 1.0
    view.CameraParallelScale = 54.99504523136608
    # create new layout object 'Layout #1'
    layout1 = CreateLayout(name="Layout #1")
    layout1.AssignView(0, view)
    layout1.SetSize(824, 656)

    # restore active view
    SetActiveView(view)

    return view

def read_from_fides(view, num_iterations):
    global fides
    cnt = 0
    while cnt < num_iterations:
        status = NotReady
        while status == NotReady:
            # must call PrepareNextStep to get Fides ready to read the
            # next step
            fides.PrepareNextStep()
            fides.UpdatePipelineInformation()
            status = fides.NextStepStatus


        print("PREP THE STEP")

        #merged_source = MergeBlocks(fides)
        #data = paraview.servermanager.Fetch(merged_source)

#        vtk_data = servermanager.Fetch(fides)
#        part_set = vtk_data.GetPartitionedDataSet(0)
#        dataset = part_set.GetPartition(0)

        # Use ParaView's algorithm output to get the local data partition directly
        fides.UpdatePipeline()

        print("UPDATE PIPELINE COMPLETE") 
        # Natively extract the local rank's data without servermanager cross-contamination
        client_side_data = fides.GetClientSideObject()
        vtk_data = client_side_data.GetOutputDataObject(0)
        
        # Pull the dataset slice belonging to THIS process
        if vtk_data.IsA("vtkPartitionedDataSetCollection") or vtk_data.IsA("vtkPartitionedDataSet"):
            part_set = vtk_data.GetPartitionedDataSet(0)
            dataset = part_set.GetPartition(0)
        else:
            dataset = vtk_data
            
        cellData = dataset.GetCellData()
        print("RANK {}, NUM CELLS: {}".format(myRank_, dataset.GetNumberOfCells()))

        #print("RANK {}, NUM CELLS:".format(myRank_) + str(dataset.GetNumberOfCells()))

        #point_data = data.GetPointData()

        pressure = numpy_support.vtk_to_numpy(cellData.GetArray("pressure"))

        print("RANK {}, PRESSURE, NUM VALS: {}".format(myRank_, len(pressure)))
        print(pressure)


        print("OUT OF HERE")
        smooth_points = CellDatatoPointData(Input=fides)

        #pipeline.UpdatePipeline()
        gridDisplay = Show(smooth_points, view, 'UnstructuredGridRepresentation')

        ColorBy(gridDisplay, ('POINTS', 'velocity'))

        gridDisplay.Representation = 'Surface'

        velocityLUT = GetColorTransferFunction('velocity')
        velocityLUT.RescaleTransferFunction(0.0 , 640.0)


        velocityPWF = GetOpacityTransferFunction('velocity')
        velocityPWF.RescaleTransferFunction(0.0, 640.0)

        gridDisplay.LookupTable = velocityLUT


        # get color legend/bar for velocityLUT in view view
        velocityLUTColorBar = GetScalarBar(velocityLUT, view)
        velocityLUTColorBar.Title = 'velocity'
        velocityLUTColorBar.ComponentTitle = 'Magnitude'
        velocityLUTColorBar.UseCustomLabels = 0

        # show color legend
        gridDisplay.SetScalarBarVisibility(view, True)
        velocityLUT.AutomaticRescaleRangeMode = 'Never'

        SetActiveView(view)
        # create extractor
        if not os.path.isdir("./catalyst_renders"):
            os.mkdir("./catalyst_renders")

        output = f"./catalyst_renders/output-{cnt:05d}.png"
        SaveScreenshot(output, view, ImageResolution=[800, 800])

        cnt+=1
    # Gather the actual arrays from here
def make_features(u_plus, du_dy, nu_t):
    f0 = (u_plus / 20.0).astype(np.float32)
    f1 = (np.abs(du_dy) * RE_TAU / 20.0).astype(np.float32)
    f2 = (np.log1p(nu_t) / 4.0).astype(np.float32)
    return np.stack([f0, f1, f2], axis=1)

if __name__ == "__main__":

    # * Accept an input arg for steps/iterations

    num_iterations = 1
    if len(sys.argv) > 1:
        num_iterations = int(sys.argv[1])

    view = SetupRenderView()
    fides.PrepareNextStep()
    fides.UpdatePipelineInformation()
    #pipeline, display = SetupVisPipelineVel(fides, view)
    comm = MPI.COMM_WORLD
    read_from_fides(view, num_iterations)
    comm.Barrier()
