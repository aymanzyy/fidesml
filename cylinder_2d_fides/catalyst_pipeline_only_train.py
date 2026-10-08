# catalyst_pipeline_cylinder.py - Catalyst rendering for cylinder flow.
# Shows omega (vorticity) field: coarse and ROM side by side.
import sys
import os
import time

from paraview.simple import *
from paraview import catalyst

from paraview.vtk.numpy_interface import dataset_adapter as dsa

from vtk.util import numpy_support
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkUnstructuredGrid
import vtkmodules.vtkCommonDataModel as vtk
print("executing catalyst_pipeline_cylinder", flush=True)

NotReady = 1
EndOfStream = 2

xml_plugin_path = os.path.abspath("/home/local/KHQ/ayman.yousef/Downloads/paraview_new_fides_writer/paraview/Remoting/Application/Resources/proxies_fides.xml")
LoadPlugin(xml_plugin_path, ns=globals())


#producer = TrivialProducer(registrationName="grid")
fides_write = FidesWriter()
fides_write.FileName = "/home/local/KHQ/ayman.yousef/Downloads/catalyst-ml/catalyst-ml/examples/2d_cyl.gp"
fides_write.Engine = "SST"

#fides_write.FileName = "/home/local/KHQ/ayman.yousef/Downloads/catalyst-ml/catalyst-ml/examples/training_service.bp"
#fides_write.Engine = "BP"

producer = TrivialProducer(registrationName="grid")
fides_write.Input = producer

# READER TO INGEST LOSS INFORMATION

#fides = FidesReader(FileName="./inference_service.bp")
fides = FidesReader()
fides.DataSourceEngines = ["source", "BPFile"]
fides.StreamSteps = 1

omega_rom = None
_result_loss = None

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

def visualize():
    # Omega coarse
    rv1 = make_view()
    d1 = Show(producer, rv1)
    d1.Representation = 'Surface'
    ColorBy(d1, ('POINTS', 'omega_coarse'))
    lut1 = GetColorTransferFunction('omega_coarse')
    lut1.ApplyPreset('Cool to Warm', True)
    lut1.RescaleTransferFunction(-0.03, 0.03)
    d1.SetScalarBarVisibility(rv1, True)
    GetScalarBar(lut1, rv1).Title = 'omega (coarse)'

    annot1 = AnnotateTimeFilter(registrationName='step_label1', Input=producer)
    annot1.Format = 'Step {time:.0f}'
    ad1 = Show(annot1, rv1)
    ad1.FontSize = 24
    ad1.WindowLocation = 'Upper Center'
    ad1.Color = [0, 0, 0]

    SetActiveView(rv1)
    p1 = CreateExtractor('PNG', rv1, registrationName='PNG_omega_coarse')
    p1.Trigger = 'TimeStep'
    p1.Writer.FileName = 'omega_coarse_{timestep:04d}.png'
    p1.Writer.ImageResolution = [1200, 400]
    p1.Writer.Format = 'PNG'

    # Omega ROM
    rv2 = make_view()
    d2 = Show(producer, rv2)
    d2.Representation = 'Surface'
    ColorBy(d2, ('POINTS', 'omega_rom'))
    lut2 = GetColorTransferFunction('omega_rom')
    lut2.ApplyPreset('Cool to Warm', True)
    lut2.RescaleTransferFunction(-0.03, 0.03)
    d2.SetScalarBarVisibility(rv2, True)
    GetScalarBar(lut2, rv2).Title = 'omega (ROM)'

    annot2 = AnnotateTimeFilter(registrationName='step_label2', Input=producer)
    annot2.Format = 'Step {time:.0f}'
    ad2 = Show(annot2, rv2)
    ad2.FontSize = 24
    ad2.WindowLocation = 'Upper Center'
    ad2.Color = [0, 0, 0]

    SetActiveView(rv2)
    p2 = CreateExtractor('PNG', rv2, registrationName='PNG_omega_rom')
    p2.Trigger = 'TimeStep'
    p2.Writer.FileName = 'omega_rom_{timestep:04d}.png'
    p2.Writer.ImageResolution = [1200, 400]
    p2.Writer.Format = 'PNG'

    options = catalyst.Options()
    options.ExtractsOutputDirectory = 'catalyst_output'
    options.GlobalTrigger.Frequency = 1

def catalyst_execute(info):
    global producer, fides_write, _result

    # Update producer to perform the correct write
    print("  [catalyst] step={:.0f}".format(info.time), flush=True)


    print(" @@@@@@@@@@@@@@@@@@@@@@@ FIDES WRITE @@@@@@@@@@@@@@@@@@@@@@@")
    producer.UpdatePipeline()
    fides_write.UpdatePipeline()

    # Update Reader to get the correct inferred value
    #time.sleep(5)

    #if (int(info.time) == 10050):
    #    fides.FileName = "./training_service.bp"
    #    fides.PrepareNextStep()
    #    fides.UpdatePipelineInformation()
    #print(" @@@@@@@@@@@@@@@@@@@@@@@ FIDES READ @@@@@@@@@@@@@@@@@@@@@@@")
    #status = NotReady
    #while status == NotReady:
        # must call PrepareNextStep to get Fides ready to read the
        # next step
    #    fides.PrepareNextStep()
    #    fides.UpdatePipelineInformation()
    #    status = fides.NextStepStatus

    #fides.UpdatePipeline()
    #merged_source = MergeBlocks(fides)

    #data = paraview.servermanager.Fetch(merged_source)
    #point_data = data.GetPointData()

    #training_loss = numpy_support.vtk_to_numpy(point_data.GetArray("training_loss"))   
    #_result_loss = training_loss if training_loss is not None else None

def catalyst_results(info):

    global _result_loss
    if _result_loss is None:
        return
    info.catalyst_params["omega_rom"] = _result_loss.tolist()
    visualize()
    print(" ******************* RESULT (TRAINING LOSS) *************************")
    print(_result_loss)
    #print(f"[CATALYST CHANNEL ML] catalyst_results: wrote omega_rom "
    #      f"max={float(max(abs(v) for v in _result)):.4f}", flush=True)
