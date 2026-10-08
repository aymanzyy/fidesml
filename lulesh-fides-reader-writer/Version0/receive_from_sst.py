# SPDX-FileCopyrightText: 2026 Oak Ridge National Laboratory and Contributors
#
# SPDX-License-Identifier: Apache-2.0

import argparse
from paraview.simple import *

from paraview import print_info

# ------------------------------------------------------------------------------
# Catalyst options
from paraview import catalyst
import sys
options = catalyst.Options()
options.GlobalTrigger = "TimeStep"
options.EnableCatalystLive = 1
options.CatalystLiveTrigger = "TimeStep"
options.ExtractsOutputDirectory = "/tmp"
# options.ExtractsOutputDirectory = '.'
#import fides
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
    producer = TrivialProducer(registrationName="fides")
    return producer


# this returns the fides proxy which should be used for post hoc vis
# i.e., you're reading .bp files
def SetupFidesReader(json, bp, sst):

    print("WRITER SETUP?")
    # ! temp, testing if FidesWriter is possible
    # ! Delete after


    # * Don't think it supports an SST engine yet
    fides_write = FidesWriter()
    #fides_write.SetFileName("/home/local/KHQ/ayman.yousef/Downloads/ayman_internship_work/ayman_internship_work/test_write.gp")
    fides_write.FileName = "/home/local/KHQ/ayman.yousef/Downloads/ayman_internship_work/ayman_internship_work/test_write.gp"
    #fides_write.SetInputConnection(producer) ## * From the XML file "The input filter/source whose output dataset is to written to the file."
    producer = TrivialProducer(registrationName="fides")

    fides_write.Input = producer
    #fides_write.ChooseArraysToWrite
    #fides_write

    print("SETTING UP FIDES JSON READER")

    fides= FidesJSONReader()
    fides.DataSourceEngines = ["source", "SST"]
    fides.DataSourcePath = ["source", "/home/local/KHQ/ayman.yousef/Downloads/ayman_internship_work/ayman_internship_work/lulesh-in-transit/gs.bp"]
    fides.StreamSteps = 1
    fides.FileName = json
    #fides = FidesJSONReader(StreamSteps=1, FileName=json)

    print("FINISHED SETUP FIDES JSON READER")

    # 'source' is the name of the ADIOS data source in the JSON data model
    # required to update the fides reader
    fides.UpdatePipelineInformation()
    return fides


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

    print("AHHHHhdhdhdhddhdh")
    print_info("in '%s::catalyst_execute'", __name__)
    global pipeline, display
    pipeline.UpdatePipeline()
    display.RescaleTransferFunctionToDataRange()

    print_info("executing (cycle={}, time={})".format(info.cycle, info.time))
    print_info("U-range: {}".format(producer.PointData["U"].GetRange(0)))
    print_info("V-range: {}".format(producer.PointData["V"].GetRange(0)))


def ParseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-j",
        "--json_filename",
        help="path to Fides JSON file",
        type=str,
        required=False,
    )
    parser.add_argument("-b", "--bp_filename", help="path to bp file", type=str, required=True)
    parser.add_argument("--staging", help="use SST engine", action="store_true")
    args = parser.parse_args()
    return args


def StreamingVis(args):
    # adios/fides step status
    NotReady = 1
    EndOfStream = 2

    # setup the reader, view, pipeline
    fides = SetupFidesReader(args.json_filename, args.bp_filename, args.staging)
    view = SetupRenderView()

    step = 0
    while True:
        status = NotReady
        while status == NotReady:
            # must call PrepareNextStep to get Fides ready to read the
            # next step
            fides.PrepareNextStep()
            fides.UpdatePipelineInformation()
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


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    print("in __main__()")
    args = ParseArgs()
    StreamingVis(args)