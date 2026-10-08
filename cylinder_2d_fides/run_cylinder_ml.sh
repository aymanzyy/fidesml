#!/bin/bash
set -e
cd "$(dirname "$0")"

source ../../.venv/bin/activate
export LD_LIBRARY_PATH=/home/local/KHQ/ayman.yousef/Downloads/paraview_build/lib:$LD_LIBRARY_PATH
export CATALYST_IMPLEMENTATION_PATHS=/home/local/KHQ/ayman.yousef/Downloads/paraview_build/lib/catalyst
CATALYST_BUILD=/home/local/KHQ/ayman.yousef/Downloads/catalyst_install/catalyst_install
CATALYST_ML=~/Downloads/catalyst-ml/catalyst-ml/
export PYTHONPATH="${CATALYST_BUILD}/lib/python3.10/site-packages:${CATALYST_ML}:${PYTHONPATH}"

export DISPLAY=:0

redis-cli FLUSHDB > /dev/null 2>&1 # 2>&1 combines error and output logs
echo "[run] Redis flushed"

rm -rf catalyst_output
mkdir -p catalyst_output

echo "[run] Starting inference service..."
python3 cylinder_inference_service.py --max-steps 100 > /tmp/infer.log 2>&1 &
INFER_PID=$! # * Picks up the process id for the thread running this
sleep 3

echo "[run] Starting simulation (async Catalyst)..."
python3 cylinder_simulation.py 2>&1
SIM_EXIT=$? # * Save the exit status

wait $INFER_PID 2>/dev/null || true # * Waits on inference to be finished,

echo ""
echo "=== Done (exit=$SIM_EXIT) ==="
echo "Images: $(ls catalyst_output/*.png 2>/dev/null | wc -l)"
