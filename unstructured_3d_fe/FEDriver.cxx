// SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
// SPDX-License-Identifier: BSD-3-Clause
#include "FEDataStructures.h"
#include <mpi.h>

#include <iostream>
#include <string>
#include <fstream>
#include <chrono>
#include <thread> // For demonstration sleep

#ifdef USE_CATALYST
#include "CatalystAdaptor.h"
#endif

// Example of a C++ adaptor for a simulation code
// where the simulation code has a axis-aligned,
// fixed topology grid. Note that through configuration
// that the driver can be run without linking
// to Catalyst.

int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
  Grid grid;
  unsigned int numPoints[3] = { 100, 100, 100};
  double spacing[3] = { 1, 1.1, 1.3 };
  grid.Initialize(numPoints, spacing);
  Attributes attributes;
  attributes.Initialize(&grid);
  
  int mpiSize = 1;
  int mpiRank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

  std::string filename = "timing_rank_" + std::to_string(mpiRank) + ".txt";
  std::vector<double> timers;
#ifdef USE_CATALYST
  CatalystAdaptor::Initialize(argc, argv);
  filename.insert(0, "cat_version_");
#endif
  unsigned int numberOfTimeSteps = 1000;

  std::ofstream output_file(filename);

  for (unsigned int timeStep = 0; timeStep < numberOfTimeSteps; timeStep++)
  {
    // use a time step length of 0.1    
    auto start = std::chrono::high_resolution_clock::now();
    double time = timeStep * 0.1;
    attributes.UpdateFields(time);


#ifdef USE_CATALYST
    CatalystAdaptor::Execute(timeStep, time, grid, attributes);
#endif
  
 auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    timers.push_back(elapsed.count());
}


#ifdef USE_CATALYST
  CatalystAdaptor::Finalize();
#endif

  // * IO after so as to not mess with any of the timings

  for (unsigned int timeStep = 0; timeStep < numberOfTimeSteps; timeStep++)
  {
    output_file << "timestep " << timeStep << "," << timers[timeStep] << "\n";
  }

  if (mpiRank == 0) {
    std::cout << "SIM DONE" << std::endl;
  }
  MPI_Finalize();
  return EXIT_SUCCESS;
}
