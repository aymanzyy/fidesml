# fidesml
Fides In-Transit ML work, as part of IPDPS 2027 submission

## 1D Flow Solver

Details: 
- Experiment Overview: Measure the runtimes across both producer/solver side of things and consumer/model side of things
- Directories
  - For Fides work: channel_flow_1d_fides 
  - For Redis work: channel_flow_1d_redis
 
- Scripts:
  -  Redis Training
    - Catalyst Adaptor class: channel_adaptor.py
  	- Combined runscript for running solver/model-side: inference_run.sh
  	- Combined runscript for running solver/model-side: train_run.sh
  	- Multilayer perceptron model: model.py
  	- Solver-side pipeline script for training and inference, write data to Redis: catalyst_channel_ml.py
  	- Standalone model training script: run_training.py
  	- Standalone model inference script: run_inference.py

  - Fides Training
    - Config file for setting blocking execution: adios2_prac_config.xml
    - Solver-side pipeline script for training and inference, write data to ADIOS2: catalyst_channel_ml_fides.py
    - Definition of Catalyst's core functions (for BP engine): catalyst_channel_ml_fides_bp.py
    - Catalyst Adaptor class: channel_adaptor.py
    - Solver-side runscript, initiate inference execution: fides_inference_run.sh
    - Model-side runscript, execute inference: fides_run_inference.sh
    - Model-side runscript, execute model training: fides_run_training.sh
    - Solver-side runscript, initiate training execution: fides_train_run.sh
    - JSON file describing data schema: flow.json
    - JSON file describing data schema: flow_bp.json
    - Standalone model inference script (with optional passback to solver): run_inference_fides.py
    - Standalone model training script: run_training_fides.py
    - Multilayer perceptron model: model.py
 - XML for Setting the Blocking Config: adios2_prac_config.xml
   
## 2D LBM Proxy

Details: 
- Experiment Overview: Train and visualize auto encoder model progress throughout the course of the training process. Medium is Jupyter notebooks, with a mix of Trame and non-Trame deployment. 
- Solver Description: Unsteady flow past a cylinder using a D2Q9 BGK Lattice Boltzmann solver on GPU (PyTorch/CUDA).
- Directory: cylinder_2d_fides 

- Scripts: 
  - Fides Training:
    - Catalyst pipeline script for model training with visualization: catalyst_pipeline_cylinder.py
    - Catalyst pipeline script for model training without visualization: catalyst_pipeline_only_train.py
    - Driver script for the solver, pulling the definition from cylinder_solver.py script: cylinder_simulation.py
    - JSON file describing data schema: flow.json
    - Solver script defining 2D cylinder class: cylinder_solver.py
    - Point-cloud auto encoder model: pn_autoencoder.py
    - Runscript for running solver for model training example: run_for_train.sh
    - Runscript for running standalone model inference script: run_inference.sh
    - Runscript for running standalone model training script: run_training.sh

  - Jupyter Notebooks:
    - For baseline visualization of training and model prediction: 2d_cylinder_train_and_vis_model_weights.ipynb
    - For training with Trame visualization: trame_train_and_vis.ipynb 
    - For a solver steering example: steer_model_train_and_query.ipynb
    - For model LR change: change_auto_encoder_lr.ipynb
  

## 3D FE Solver

Details: 
- Experiment Overview: Run experiments with data written to stager (or disk in case of BP) and measure on the writer side how long it takes across 1000 timesteps and 4 ranks. 
- Solver Description: The solver is a Finite Element (FE) based solver, creating an example mesh and updating mesh fields (velocity, pressure) iteratively throughout the lifetime of the simulation. Field values are dependent on the time field.
- Directory: unstructured_3d_fe
  
- Scripts:
  - Definition of the Catalyst Adaptor class: CatalystAdaptor.h
  - Definition of data structures for solver: FEDataStructures.cxx/.h
  - Driver code for 3D FE solver: FEDriver.cxx
  - JSON file describing data schema: flow_unstructured.json
  - Runscript for running solver without Catalyst enabled: no_cat_run_sim.sh
  - Point-cloud autoencoder model running on multiple ranks: pn_autoencoder_multi_rank.py
  - Runscript for running solver with Catalyst enabled (SST engine): with_cat_run_sim.sh
  - Runscript for running solver with Catalyst enabled (BP engine): with_cat_run_sim_bp.sh
  - Standalone Python script for model training, placeholder (SST): write_sst_non_blocking.py
  - Standalone Python script for model training, placeholder (BP): write_bp_non_blocking.py
  - Catalyst Pipelinescript accompanying solver execution (SST): write_to_fides.py
  - Catalyst Pipelinescript accompanying solver execution (BP): write_to_fides_bp.py
    

## LULESH
Details: 
- Experiment Overview: Instrument in-transit visualization to exercise the ParaView/Fides/Catalyst/ADIOS2 piping
- Solver Description: Proxy-application for LLNL's production ALE3D code that models the Sedov blast problem.
- Directory:

- Scripts:
  - Runscript for running analysis endpoint: reader_run.sh
  - Runscript for running solver endpoint: writer_run.sh
  - Example compile script: rebuild.sh
  - LULESH Communication code: Version0/lulesh-comm.cc
  - LULESH Initialization code: Version0/lulesh-init.cc
  - LULESH Visualization code: Version0/lulesh-viz.cc
  - Main LUELSH driver code: Version0/lulesh.cc/.h
  - Standalone Python script for pulling data from ADIOS2 via Fides: read_from_fides.py
  - Standalone Python script for pulling data from ADIOS2 via Fides with BP engine: read_from_fides_bp.py
  - Catalyst Pipeline script for rendering: reader_create_image.py
  - Catalyst Pipeline script to write data to ADIOS2 via Fides: write_to_fides.py
  - Catalyst Pipeline script to write data to ADIOS2 via Fides (BP engine): write_to_fides_bp.py
