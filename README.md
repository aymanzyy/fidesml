# fidesml
Fides In-Transit ML work, as part of IPDPS 2027 submission

## 1D Flow Solver

Details: 
- Experiment Overview: Measure the runtimes across both producer/solver side of things and consumer/model side of things
- Directory:
  - For Fides work: channel_flow_1d_fides 
  - For Redis work: channel_flow_1d_redis
 
- Scripts:
  -  Redis Training: 
  	- inference_run.sh
  	- train_run.sh
  	- catalyst_channel_ml.py
  	- run_training.py
  	- run_inference.py
  - Fides Training:
  	- train_run.sh
  	- inference_run.sh
  	- catalyst_channel_ml_fides.py
 - XML for Setting the Blocking Config: adios2_prac_config.xml
   
## 2D LBM Proxy

Details: 
- Experiment Overview: Train and visualize auto encoder model progress throughout the course of the training process. Medium is Jupyter notebooks, with a mix of Trame and non-Trame deployment. 
- Solver Description: Unsteady flow past a cylinder using a D2Q9 BGK Lattice Boltzmann solver on GPU (PyTorch/CUDA).
- Directory: cylinder_2d_fides 

- Scripts: 
  - Redis Training: 
  	- inference_run.sh
  	- train_run.sh
  	- catalyst_channel_ml.py
  	- run_training.py
  	- run_inference.py
  - Fides Training:
  	- train_run.sh
  	- inference_run.sh
  	- catalyst_channel_ml_fides.py

- Jupyter Notebooks:
  - For baseline visualization of training and model prediction: 2d_cylinder_train_and_vis_model_weights.ipynb
  - For training with Trame visualization: trame_train_and_vis.ipynb 
  - For steering: steer_model_train_and_query.ipynb
  - For model LR change: change_auto_encoder_lr.ipynb
  

## 3D FE Solver

Details: 
- Experiment Overview: Run experiments with data written to stager (or disk in case of BP) and measure on the writer side how long it takes across 1000 timesteps and 4 ranks. 
- Solver Description: The solver is a Finite Element (FE) based solver, creating an example mesh and updating mesh fields (velocity, pressure) iteratively throughout the lifetime of the simulation. Field values are dependent on the time field.
- Directory: unstructured_3d_fe 
- Scripts:
  - run_training.sh
  - run_for_return_train.sh
  - catalyst_pipeline_train_with_return.py
  - cylinder_simulation.py
  - catalyst_train_service.py 

 ## LULESH
