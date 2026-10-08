"""
catalyst_channel_ml.py — Catalyst V2 script for channel flow ML coupling.

KEY CATALYST V2 FACTS (learned from debugging):
  1. catalyst_initialize() is called with NO arguments by Catalyst V2.
     Init params MUST be read here via GetCatalystParameters(), which
     returns the initialization node (including catalyst_params/ sub-tree).

  2. catalyst_execute(info).catalyst_params is the EXECUTE node only:
     catalyst/state/timestep, catalyst/channels/... are present but
     catalyst_params/ is NOT — that tree is only in the init node.

  3. set_external() holds a raw pointer. Always store typed arrays in named
     variables (not temporaries like arr.astype(...)) before passing to
     set_external, or the array can be GC'd before execute() reads it.
"""
print("[CATALYST CHANNEL ML] loading...", flush=True)

import sys
import numpy as np
from paraview.simple import *
from paraview.simple import TrivialProducer
from paraview.catalyst import Options
import os
from paraview.modules.vtkPVInSitu import vtkInSituPythonConduitHelper
from paraview.vtk.numpy_interface import dataset_adapter as dsa
from vtk.util import numpy_support

import time
from pathlib import Path

options = Options()

_transport = None
_mode      = "inference"
_Re_tau    = 180.0
_result    = None


NotReady = 1
EndOfStream = 2


xml_plugin_path = os.path.abspath("/home/local/KHQ/ayman.yousef/Downloads/paraview_new_fides_writer/paraview/Remoting/Application/Resources/proxies_fides.xml")
LoadPlugin(xml_plugin_path, ns=globals())


#producer = TrivialProducer(registrationName="grid")
fides_write = FidesWriter()
fides_write.FileName = "/home/local/KHQ/ayman.yousef/Downloads/ayman_internship_work/ayman_internship_work/test_write.gp"
fides_write.Engine = "SST"
producer = TrivialProducer(registrationName="grid")
fides_write.Input = producer

def _get(node, path, default=""):
    """Safe conduit accessor — missing paths return empty node, str('') == ''."""
    try:
        v = str(node[path])
        return v if v else default
    except Exception:
        return default


def catalyst_initialize():
    """Called by Catalyst V2 with NO arguments.
    Read init params here — they are NOT available in catalyst_execute."""
    global _transport, _mode, _Re_tau

    params   = vtkInSituPythonConduitHelper.GetCatalystParameters()
    _mode    = _get(params, "catalyst_params/catalyst/ml_coupling/mode",    "inference")

    print("[CATALYST CHANNEL ML] catalyst_initialize called", flush=True)

    # GetCatalystParameters() in init context returns the full init node,
    # including the catalyst_params/ sub-tree set in channel_adaptor.initialize()

    print(f"[CATALYST CHANNEL ML] initialized mode={_mode} Re_tau={_Re_tau}",
          flush=True)


def catalyst_execute(info):
    global _transport, _mode, _Re_tau, _result, fides_write

    producer.UpdatePipeline()
    params = info.catalyst_params   # execute node: has catalyst/state/... only

    fides_write.UpdatePipeline()
    try:
        step   = int(params["catalyst/state/timestep"])
        fields = params["catalyst/channels/grid/data/fields"]
        feat_dict = {
            "y_plus": np.array(fields["y_plus/values"]),
            "u_plus": np.array(fields["u_plus/values"]),
            "du_dy":  np.array(fields["du_dy/values"]),
            "nu_t":   np.array(fields["nu_t/values"]),
        }
    except Exception as e:
        sys.stderr.write(f"[CATALYST CHANNEL ML] field access failed: {e}\n")
        return

    if step % 50 == 0:
        print(f"[CATALYST CHANNEL ML] step={step}", flush=True)

    if _mode == "inference":


        file_path = Path("./inference_service.bp")

        # Loop runs as long as the file does not exist
        while not file_path.exists():
            print("Waiting for file...")
            time.sleep(1)  # Check every 1 second

        fides = FidesReader(FileName="./inference_service.bp")
        fides.DataSourceEngines = ["source", "BPFile"]
        fides.StreamSteps = 1
        fides.PrepareNextStep()
        fides.UpdatePipelineInformation()

        status = NotReady
        while status == NotReady:
            # must call PrepareNextStep to get Fides ready to read the
            # next step
            fides.PrepareNextStep()
            fides.UpdatePipelineInformation()
            status = fides.NextStepStatus
        fides.UpdatePipeline()
        merged_source = MergeBlocks(fides)

        data = paraview.servermanager.Fetch(merged_source)
        point_data = data.GetPointData()


        delta_nu_t = numpy_support.vtk_to_numpy(point_data.GetArray("delta_nu_t"))   

        print("INFERRED DELTA_NU_T IS:")
        print(delta_nu_t)
        #result  = _transport.wait_for_result(rank=0, step=step)
        # Store correction — catalyst_results() will write it into
        # info.catalyst_params for the adaptor to read via catalyst.results()
        _result = delta_nu_t if delta_nu_t is not None else None



def catalyst_results(info):
    """Write correction back to the adaptor via info.catalyst_params.

    Called by Catalyst immediately after catalyst_execute. The adaptor reads
    this via catalyst.results(node) — see channel_adaptor.coprocess().
    info.catalyst_params here is the OUTPUT node, not the execute input node.
    """
    global _result
    if _result is None or _mode != "inference":
        return
    info.catalyst_params["delta_nu_t"] = _result.tolist()
    print(f"[CATALYST CHANNEL ML] catalyst_results: wrote delta_nu_t "
          f"max={float(max(abs(v) for v in _result)):.4f}", flush=True)


print("[CATALYST CHANNEL ML] ready", flush=True)
