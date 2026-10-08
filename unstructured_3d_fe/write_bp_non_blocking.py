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

xml_plugin_path = os.path.abspath("/home/local/KHQ/ayman.yousef/Downloads/paraview_with_qt/paraview/Remoting/Application/Resources/proxies_fides.xml")

LoadPlugin(xml_plugin_path, ns=globals())

options = catalyst.Options()
options.GlobalTrigger = "TimeStep"
options.EnableCatalystLive = 1
options.CatalystLiveTrigger = "TimeStep"
options.ExtractsOutputDirectory = "../lulesh_v0_build/images/"
# options.ExtractsOutputDirectory = '.'

DISABLE_EXTRACTOR = False

## * Set up the writer in a global context?


# * Don't think it supports an SST engine yet
fides_write = FidesWriter()
fides_write.FileName = "./sst_test_write.bp"
fides_write.Engine = 0
#fides_write.AdiosConfigFile = "/home/local/KHQ/ayman.yousef/Downloads/CatalystExamples/catalyst-examples/ParaView/UnstructuredFidesML/sst_no_block.xml"

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


def SetupFidesWriter(producer):
    global fides_write

    fides_write.Input = producer
    return fides_write



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
    ## * Update the writer to write
    global fides_write
    fides_write.UpdatePipeline()

    print_info("executing (cycle={}, time={})".format(info.cycle, info.time))
    #print_info("U-range: {}".format(producer.PointData["U"].GetRange(0)))
    #print_info("V-range: {}".format(producer.PointData["V"].GetRange(0)))


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # in this case we're running from Catalyst
    print("in __main__()")
else:
    view = SetupRenderView()
    producer = SetupCatalystProducer()
    writer = SetupFidesWriter(producer)