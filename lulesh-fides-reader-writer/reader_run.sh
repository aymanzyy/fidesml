export ADIOS2_PLUGIN_PATH=/home/local/KHQ/ayman.yousef/.local/lib
export CATALYST_IMPLEMENTATION_NAME=paraview
export CATALYST_IMPLEMENTATION_PATHS=/home/local/KHQ/ayman.yousef/Downloads/paraview_new_fides_writer/paraview_build/lib/catalyst

mpirun -n 1 ./myLulesh-build/lulesh2.0 -s 10 -i 100 -w reader -k ./Version0/read_from_fides.py -p 
