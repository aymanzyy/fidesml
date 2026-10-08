#!/bin/sh
# run.sh — run the channel flow 1D catalyst-ml example
#
# Usage:
#   sh run.sh training    run sim in training mode (pair with run_training.py)
#   sh run.sh inference   run sim in inference mode (pair with run_inference.py)
#
# Prerequisites:
#   redis-server &
#   pvpython reference.py --output /abs/path/to/channel_ref.npz
#   python run_training.py --ref /abs/path/to/channel_ref.npz  (training only)
#   python run_inference.py --model /abs/path/to/channel_correction.onnx (inference only)

MODE=${1:-inference}

PARAVIEW_BUILD=/Users/jeff.lee/gitsync/paraview_build
CATALYST_BUILD=/Users/jeff.lee/gitsync/catalyst_build
CATALYST_ML=/Users/jeff.lee/gitsync/catalyst-ml

PYTORCH_SITE=/opt/homebrew/Cellar/pytorch/2.11.0/libexec/lib/python3.14/site-packages

export DYLD_LIBRARY_PATH="${PARAVIEW_BUILD}/lib:${DYLD_LIBRARY_PATH}"
# No KMP_DUPLICATE_LIB_OK needed — brew pytorch links same OpenBLAS as numpy
export CATALYST_IMPLEMENTATION_PATHS="${PARAVIEW_BUILD}/lib/catalyst"
export CATALYST_IMPLEMENTATION_NAME=paraview
export PYTHONPATH="${CATALYST_BUILD}/lib/python3.14/site-packages:${CATALYST_ML}:${PYTHONPATH}"

# PyTorch bundles its own libomp.dylib; paraview/OpenBLAS brings another.
# Without this flag the training process crashes with __kmp_register_library_startup.
export KMP_DUPLICATE_LIB_OK=TRUE

PVPYTHON="${PARAVIEW_BUILD}/bin/pvpython"
PYTHON3="/opt/homebrew/Frameworks/Python.framework/Versions/3.14/bin/python3.14"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "[run.sh] mode=${MODE}"
"${PVPYTHON}" \
    "${SCRIPT_DIR}/channel_sim.py" \
    "${SCRIPT_DIR}/catalyst_channel_ml.py" \
    "${MODE}"
