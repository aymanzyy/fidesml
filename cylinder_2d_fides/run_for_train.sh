export ADIOS2_PLUGIN_PATH=/home/local/KHQ/ayman.yousef/.local/lib
export CATALYST_IMPLEMENTATION_NAME=paraview
export CATALYST_IMPLEMENTATION_PATHS=/home/local/KHQ/ayman.yousef/Downloads/paraview_new_fides_writer/paraview_build/lib/catalyst

PARAVIEW_BUILD=/home/local/KHQ/ayman.yousef/Downloads/paraview_new_fides_writer/paraview_build
CATALYST_BUILD=/home/local/KHQ/ayman.yousef/Downloads/catalyst_install/catalyst_install

CATALYST_ML=~/Downloads/catalyst-ml/catalyst-ml/
#export PYTHONPATH=$PYTHONPATH:/home/local/KHQ/ayman.yousef/Downloads/catalyst_install/catalyst_install/lib/python3.10/site-packages/

export PYTHONPATH="${CATALYST_BUILD}/lib/python3.10/site-packages:${CATALYST_ML}:${PYTHONPATH}"

PVPYTHON="${PARAVIEW_BUILD}/bin/pvpython"

"${PVPYTHON}" --force-offscreen-rendering cylinder_simulation.py catalyst_pipeline_only_train.py train
