rm -r build 
export ADIOS2_PLUGIN_PATH=/usr/local/lib
export CATALYST_IMPLEMENTATION_NAME=paraview
export CATALYST_IMPLEMENTATION_PATHS=/home/local/KHQ/ayman.yousef/Downloads/paraview_with_qt/paraview_build/lib/catalyst
export catalyst_DIR=/home/local/KHQ/ayman.yousef/Downloads/catalyst_install/catalyst_install/lib/cmake/catalyst-2.1
export PYTHONPATH="/home/local/KHQ/ayman.yousef/Downloads/paraview_with_qt/paraview_build/lib/python3.10/site-packages:$PYTHONPATH"
export PYTHONPATH="/home/local/KHQ/ayman.yousef/Downloads/catalyst_install/catalyst_install/lib/python3.10/site-packages:$PYTHONPATH"

PARAVIEW_BUILD=/home/local/KHQ/ayman.yousef/Downloads/paraview_with_qt/
CATALYST_BUILD=/home/local/KHQ/ayman.yousef/Downloads/catalyst_install/catalyst_install

export PYTHONPATH="${CATALYST_BUILD}/lib/python3.10/site-packages:${CATALYST_ML}:${PYTHONPATH}"

PVPYTHON="${PARAVIEW_BUILD}/bin/pvpython"

cmake -S . -B build -DParaView_CATALYST_DIR=/home/local/KHQ/ayman.yousef/Downloads/paraview_with_qt/paraview_build/lib/catalyst

mkdir -p build

cmake --build build

mpirun -np 4 ./build/bin/UnstructuredGridML write_sst_non_blocking.py 
