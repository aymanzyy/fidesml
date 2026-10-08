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
from paraview.simple import TrivialProducer
from paraview.catalyst import Options

options = Options()

_transport = None
_mode      = "inference"
_Re_tau    = 180.0
_result    = None

producer = TrivialProducer(registrationName="grid")


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

    print("[CATALYST CHANNEL ML] catalyst_initialize called", flush=True)

    try:
        from catalyst_ml.transport import RedisTransport
        from paraview.modules.vtkPVInSitu import vtkInSituPythonConduitHelper
    except ImportError as e:
        sys.stderr.write(f"[CATALYST CHANNEL ML] ImportError: {e}\n")
        return

    # GetCatalystParameters() in init context returns the full init node,
    # including the catalyst_params/ sub-tree set in channel_adaptor.initialize()
    params   = vtkInSituPythonConduitHelper.GetCatalystParameters()
    _mode    = _get(params, "catalyst_params/catalyst/ml_coupling/mode",    "inference")
    endpoint = _get(params, "catalyst_params/catalyst/ml_coupling/redis",   "localhost:6379")
    channel  = _get(params, "catalyst_params/catalyst/ml_coupling/channel", "ml_closure")
    re_str   = _get(params, "catalyst_params/catalyst/ml_coupling/Re_tau",  "180")
    _Re_tau  = float(re_str) if re_str else 180.0

    _transport = RedisTransport(endpoint=endpoint, channel=channel)
    _transport.publish_schema({
        "version": 1, "mode": _mode, "channel": channel,
        "fields": {
            "u_plus":  {"dtype": "float64", "ncomp": 1},
            "du_dy":   {"dtype": "float64", "ncomp": 1},
            "nu_t":    {"dtype": "float64", "ncomp": 1},
            "y_plus":  {"dtype": "float64", "ncomp": 1},
        },
        "output": {"delta_nu_t": {"dtype": "float64", "ncomp": 1}},
    })
    print(f"[CATALYST CHANNEL ML] initialized mode={_mode} Re_tau={_Re_tau}",
          flush=True)


def catalyst_execute(info):
    global _transport, _mode, _Re_tau, _result

    if _transport is None:
        return

    producer.UpdatePipeline()
    params = info.catalyst_params   # execute node: has catalyst/state/... only

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

    # Reconstruct 2D coords for SpatialReferenceData lookup
    y = feat_dict["y_plus"]
    feat_dict["coords_2d"] = np.stack([y, np.zeros_like(y)], axis=1).flatten()

    _transport.push_features(rank=0, step=step, fields=feat_dict)

    if _mode == "inference":
        result  = _transport.wait_for_result(rank=0, step=step)
        # Store correction — catalyst_results() will write it into
        # info.catalyst_params for the adaptor to read via catalyst.results()
        _result = result["delta_nu_t"] if result is not None else None

    if step % 50 == 0:
        print(f"[CATALYST CHANNEL ML] step={step}", flush=True)


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
