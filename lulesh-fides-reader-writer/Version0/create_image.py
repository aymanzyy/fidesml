# SPDX-FileCopyrightText: 2026 Oak Ridge National Laboratory and Contributors
#
# SPDX-License-Identifier: Apache-2.0

import argparse
from paraview.simple import *

from paraview import print_info

# ------------------------------------------------------------------------------
# Catalyst options
from paraview import catalyst

options = catalyst.Options()
options.GlobalTrigger = "TimeStep"
options.EnableCatalystLive = 1
options.CatalystLiveTrigger = "TimeStep"
options.ExtractsOutputDirectory = "../lulesh_v0_build/images/"
# options.ExtractsOutputDirectory = '.'

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


# takes in a producer and view and sets up the visualization pipeline
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


# Catalyst uses this to update the pipeline and print out some info on each time step
# you don't need to call this directly in your script; ParaView Catalyst will call it for you
def catalyst_execute(info):
    print_info("in '%s::catalyst_execute'", __name__)
    global pipeline, display
    pipeline.UpdatePipeline()
    display.RescaleTransferFunctionToDataRange()

    #print_info("executing (cycle={}, time={})".format(info.cycle, info.time))
    #print_info("U-range: {}".format(producer.PointData["U"].GetRange(0)))
    #print_info("V-range: {}".format(producer.PointData["V"].GetRange(0)))


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # in this case we're running from Catalyst
    print("in __main__()")
else:
    view = SetupRenderView()
    producer = SetupCatalystProducer()
    pipeline, display = SetupVisPipeline(producer, view)
    # normally not needed, but a bug fix is in progress to fix an issue
    # when using ParaView Live with extractors
    if not DISABLE_EXTRACTOR:
        SetupExtractor(view)
