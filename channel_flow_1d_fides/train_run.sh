PARAVIEW_BUILD=/home/local/KHQ/ayman.yousef/Downloads/paraview_build
CATALYST_BUILD=/home/local/KHQ/ayman.yousef/Downloads/catalyst_install/catalyst_install

CATALYST_ML=~/Downloads/catalyst-ml/catalyst-ml/

export CATALYST_IMPLEMENTATION_PATHS=/home/local/KHQ/ayman.yousef/Downloads/paraview_build/lib/catalyst
#export PYTHONPATH=$PYTHONPATH:/home/local/KHQ/ayman.yousef/Downloads/catalyst_install/catalyst_install/lib/python3.10/site-packages/

export PYTHONPATH="${CATALYST_BUILD}/lib/python3.10/site-packages:${CATALYST_ML}:${PYTHONPATH}"

PVPYTHON="${PARAVIEW_BUILD}/bin/pvpython"

redis-server & "${PVPYTHON}" channel_sim.py catalyst_channel_ml.py training & python3 run_training.py --ref channel_ref.npz --steps 500
