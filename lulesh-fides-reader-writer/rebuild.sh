cmake -G Ninja -S Version0 -B myLulesh-build -DWITH_MPI=1 -Dcatalyst_DIR=/home/local/KHQ/ayman.yousef/Downloads/catalyst_install/catalyst_install/lib/cmake/catalyst-2.1 -DPARAVIEW_CATALYST_DIR=/home/local/KHQ/ayman.yousef/Downloads/paraview_new_fides_writer/paraview_build/lib/catalyst
cmake --build myLulesh-build/
