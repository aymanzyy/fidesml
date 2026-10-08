# SPDX-FileCopyrightText: 2026 Oak Ridge National Laboratory and Contributors
#
# SPDX-License-Identifier: Apache-2.0

import argparse
from paraview.simple import *

from paraview import print_info
import os
# ------------------------------------------------------------------------------
# Catalyst options
from paraview import catalyst
import vtk

xml_plugin_path = os.path.abspath("/home/local/KHQ/ayman.yousef/Downloads/paraview_new_fides_writer/paraview/Remoting/Application/Resources/proxies_fides.xml")

LoadPlugin(xml_plugin_path, ns=globals())

options = catalyst.Options()
options.GlobalTrigger = "TimeStep"
options.EnableCatalystLive = 1
options.CatalystLiveTrigger = "TimeStep"
options.ExtractsOutputDirectory = "../lulesh_v0_build/images/"



NotReady = 1
EndOfStream = 2

DISABLE_EXTRACTOR = False

# setup and returns the view
def SetupRenderView():
    # Create a new 'Render View'
    view = CreateView("RenderView")

    camera = GetActiveCamera()
    camera.Azimuth(230)
    camera.Elevation(-45)

    SetActiveView(None)

    # create new layout object 'Layout #1'
    layout1 = CreateLayout(name="Layout #1")
    layout1.AssignView(0, view)
    layout1.SetSize(824, 656)

    # restore active view
    SetActiveView(view)
    return view


# A different proxy type is used depending on whether you're using
# catalyst, or doing post hoc visualization
# this returns the catalyst producer
def SetupCatalystProducer():
    producer = TrivialProducer(registrationName="grid")
    return producer


def SetupFidesReader():

    fides=FidesJSONReader()
    fides.DataSourceEngines = ["source", "SST"]
    fides.DataSourcePath = ["source", "/home/local/KHQ/ayman.yousef/Downloads/ayman_internship_work/ayman_internship_work/test_write.gp"]
    fides.StreamSteps = 1
    fides.FileName = "./Version0/gs-fides.json"

    fides.UpdatePipelineInformation()
    return fides



# sets up an extractor for writing out PNG images
def SetupExtractor(view):
    # create extractor
    pNG1 = CreateExtractor("PNG", view, registrationName="PNG1")
    # trace defaults for the extractor.
    pNG1.Trigger = "TimeStep"

    # init the 'PNG' selected for 'Writer'
    pNG1.Writer.FileName = "output_{timestep:06d}.png"
    pNG1.Writer.ImageResolution = [800, 800]
    pNG1.Writer.Format = "PNG"


def UpdateFides():
    fides.PrepareNextStep()
    fides.UpdatePipelineInformation()
    status = fides.NextStepStatus


# Catalyst uses this to update the pipeline and print out some info on each time step
# you don't need to call this directly in your script; ParaView Catalyst will call it for you
def catalyst_execute(info):
    print_info("in '%s::catalyst_execute'", __name__)
    #global pipeline, display
    ## * Update the writer to write
    print_info("executing (cycle={}, time={})".format(info.cycle, info.time))
    StreamingVisExecute(info.cycle)

def StreamingVisExecute(cycle):
    global pipeline, view, fides

    status = NotReady
    while status == NotReady:
        # must call PrepareNextStep to get Fides ready to read the
        # next step
        fides.PrepareNextStep()
        fides.UpdatePipelineInformation()
        status = fides.NextStepStatus
    pipeline.UpdatePipeline()
    display.RescaleTransferFunctionToDataRange()
    output = f"output-{cycle:05d}.png"
    SaveScreenshot(output, view, ImageResolution=[800, 800])

def StreamingVis(view):
    # adios/fides step status
    NotReady = 1
    EndOfStream = 2

    # setup the reader, view, pipeline
    fides = SetupFidesReader()
    step = 0
    while True:
        status = NotReady
        while status == NotReady:
            # must call PrepareNextStep to get Fides ready to read the
            # next step
            fides.PrepareNextStep()
            fides.UpdatePipelineInformation()
            info = fides.GetDataInformation()
            status = fides.NextStepStatus
        if status == EndOfStream:
            # done reading the file
            return
        if step == 0:
            # set up the pipeline on the first step
            pipeline, display = SetupVisPipeline(fides, view)

        # need to update the pipeline and then save the output
        pipeline.UpdatePipeline()
        display.RescaleTransferFunctionToDataRange()
        output = f"output-{step:05d}.png"
        SaveScreenshot(output, view, ImageResolution=[800, 800])
        step += 1

def SetupVisPipeline(producer, view):

    display_nrom = Show(producer, view, "UnstructuredGridRepresentation")

    smooth_points = CellDatatoPointData(Input=producer)
    #cellDisplay = Show(smooth_points, view, "UnstructuredGridRepresentation")

    lut = GetColorTransferFunction('velocity')

    # 2. Apply the desired preset by name
    lut.ApplyPreset('Inferno', True)

    ColorBy(display_nrom, ('POINTS', 'velocity'))

    view.ResetCamera()

    return smooth_points, display_nrom


view = SetupRenderView()
fides = SetupFidesReader()
fides.PrepareNextStep()
fides.UpdatePipelineInformation()
pipeline, display = SetupVisPipeline(fides, view)

# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # in this case we're running from Catalyst
    print("in __main__()")
else:
    # * Get's called once on import
    print("import-style")
    # * This way of doing things means we don't have to cycle through the "cycle" variable on the lulesh side
    # * Works, but I wanna make it so that Lulesh reader has to cycle through  
    # ? view = SetupRenderView()
    # ? StreamingVis(view)
    # ?if not DISABLE_EXTRACTOR:
    # ?   SetupExtractor(view)
