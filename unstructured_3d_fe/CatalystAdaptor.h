// SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
// SPDX-License-Identifier: BSD-3-Clause
#ifndef CatalystAdaptor_h
#define CatalystAdaptor_h

#include "FEDataStructures.h"
#include <catalyst.hpp>
#include <catalyst_conduit.hpp>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <mpi.h>
/**
 * The namespace hold wrappers for the three main functions of the catalyst API
 * - catalyst_initialize
 * - catalyst_execute
 * - catalyst_finalize
 * Although not required it often helps with regards to complexity to collect
 * catalyst calls under a class /namespace.
 */
namespace CatalystAdaptor
{

/**
 * In this example, we show how we can use Catalysts's C++
 * wrapper around conduit's C API to create Conduit nodes.
 * This is not required. A C++ adaptor can just as
 * conveniently use the Conduit C API to setup the
 * `conduit_node`. However, this example shows that one can
 * indeed use Catalyst's C++ API, if the developer so chooses.
 */
void Initialize(int argc, char* argv[])
{

	std::cout << "CAT INIT" << std::endl;
      
	std::string temp;
  if (argc < 2)
  {
    temp = "Usage: ";
    temp.append(argv[0]).append(" inputfile");
    throw std::runtime_error(temp);
  }
  // Use the contents of the input file to initialize Catalyst
  std::ifstream input(argv[1]);
  if (!input.is_open())
  {
    temp = "Could not open: ";
    temp.append(argv[1]);
    throw std::runtime_error(temp);
  }
  std::stringstream buffer;
  buffer << input.rdbuf();

  // Populate the catalyst_initialize argument based on the "initialize" protocol [1].
  // [1] https://docs.paraview.org/en/latest/Catalyst/blueprints.html#protocol-initialize
  //conduit_cpp::Node node;
  //conduit_node_parse(conduit_cpp::c_node(&node), buffer.str().c_str(), "yaml");
  //node.print();
/*
  for (int cc = 2; cc < argc; ++cc)
  {
    std::cout << "APPEND" << std::endl;
    conduit_cpp::Node list_entry = node["catalyst/scripts/script/args"].append();
    list_entry.set(argv[cc]);
  }
*/
  conduit_cpp::Node node;
  node["catalyst_load/implementation"]         = "paraview";
  node["catalyst_load/search_paths/paraview"]  = PARAVIEW_IMPL_DIR;
  node["catalyst/scripts/script0"].set_string(argv[1]);

  catalyst_status err = catalyst_initialize(conduit_cpp::c_node(&node));
  if (err != catalyst_status_ok)
  {
    std::cerr << "Failed to initialize Catalyst: " << err << std::endl;
  }
}

void Execute(int cycle, double time, Grid& grid, Attributes& attribs)
{
  // Populate the catalyst_execute argument based on the "execute" protocol [3].
  // [3] https://docs.paraview.org/en/latest/Catalyst/blueprints.html#protocol-execute

  conduit_cpp::Node exec_params;

  // State: Information about the current iteration. All parameters are
  // optional for catalyst but downstream filters may need them to execute
  // correctly.

  // add time/cycle information
  auto state = exec_params["catalyst/state"];
  state["timestep"].set(cycle);
  state["time"].set(time);
  state["multiblock"].set(1);

  // Channels: Named data-sources that link the data of the simulation to the
  // analysis pipeline in other words we map the simulation datastructures to
  // the ones expected by ParaView.  In this example we use the Mesh Blueprint
  // to describe data see also bellow.

  // Add channels.
  // We only have 1 channel here. Let's name it 'grid'.
  auto channel = exec_params["catalyst/channels/grid"];

  // Since this example is using Conduit Mesh Blueprint to define the mesh,
  // we set the channel's type to "mesh".
  channel["type"].set("mesh");

  // now create the mesh.
  auto mesh = channel["data"];

  // populate the data node following the Mesh Blueprint [4]
  // [4] https://llnl-conduit.readthedocs.io/en/latest/blueprint_mesh.html


  std::vector<double> coordsX;
  std::vector<double> coordsY;
  std::vector<double> coordsZ;

  auto* pointsData = grid.GetPointsArray();
  std::vector<double> pointsvec(pointsData, pointsData + grid.GetNumberOfPoints() * 3);

  std::vector<double> velocityMag;
  for (int ni=0; ni < grid.GetNumberOfPoints() ; ++ni) {
    coordsX.push_back(double(pointsvec[3 * ni + 0]));
    coordsY.push_back(double(pointsvec[3 * ni + 1]));
    coordsZ.push_back(double(pointsvec[3 * ni + 2]));
    //std::cout << "Added the following coords: " << coordsX[ni] << ", " << coordsY[ni] << ", " << coordsZ[ni] << std::endl;
  }


  std::vector<float> float_coordsX(coordsX.begin(), coordsX.end());
  std::vector<float> float_coordsY(coordsY.begin(), coordsY.end());
  std::vector<float> float_coordsZ(coordsZ.begin(), coordsZ.end());


  // * Think that connectivity is wrong, create it here


  //std::cout << "len of coordsX: " <<coordsX.size() << std::endl;

  // start with coordsets (of course, the sequence is not important, just make
  // it easier to think in this order).
  mesh["coordsets/coordinates/type"].set("explicit");
  mesh["coordsets/coordinates/values/x"].set_external(float_coordsX);
  mesh["coordsets/coordinates/values/y"].set_external(float_coordsY);
  mesh["coordsets/coordinates/values/z"].set_external(float_coordsZ);

  
  // Next, add topology
  mesh["topologies/mesh/type"] = "unstructured";
  mesh["topologies/mesh/coordset"] =  "coordinates";
  mesh["topologies/mesh/elements/shape"] = "hex";

  // * Trad way of setting up
  
  //mesh["topologies/mesh/elements/connectivity"].set_external(
  //    grid.GetCellPoints(0), grid.GetNumberOfCells() * 8);

  
  //mesh["topologies/mesh/elements/connectivity"].set_external(reinterpret_cast<int64_t*>(grid.GetCellPoints(0)), grid.GetNumberOfCells() * 8);


  // ! Non-trad way, wasn't working but worth another look

   std::vector<int64_t> connectivity_array(grid.GetCellVec()->begin(), grid.GetCellVec()->end());
  //vecPtr, vecPtr->end()
  
  /*
 for (const auto& num : connectivity_array) {
   std::cout << "connectivity: " << num << " ";
  }
  */
  mesh["topologies/mesh/elements/connectivity"].set_external(connectivity_array);

  

  // Finally, add fields.

  // First component of the path is the name of the field . The rest are described
  // in https://llnl-conduit.readthedocs.io/en/latest/blueprint_mesh.html#fields
  // under the Material-Independent Fields section.
  auto fields = mesh["fields"];


  double* vels = attribs.GetVelocityArray();

  int numPts = grid.GetNumberOfPoints();

  //int totNumPts = numPts[0] * numPts[1] * numPts[2];

  int mpiSize = 1;
  int mpiRank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

  std::cout << "rank " << mpiRank << " totNUmpts: " << numPts << std::endl;

  std::vector<double> vel_mags;

  std::vector<double> writer_ranks;

  for (int xx =0; xx < numPts; xx++) {
    vel_mags.push_back(vels[xx]);
    writer_ranks.push_back(mpiRank);
  }
  fields["velocity/association"].set("vertex");
  fields["velocity/topology"].set("mesh");
  fields["velocity/values"].set_external(vel_mags);

  fields["writer_rank/association"].set("vertex");
  fields["writer_rank/topology"].set("mesh");
  fields["writer_rank/values"].set_external(writer_ranks);

  // pressure is cell-data.
  fields["pressure/association"].set("element");
  fields["pressure/topology"].set("mesh");
  fields["pressure/volume_dependent"].set("false");
  fields["pressure/values"].set_external(attribs.GetPressureArray(), grid.GetNumberOfCells());
  
  //std::cout << "len of connectivity: " <<connectivity_array.size() << std::endl;

  catalyst_status err = catalyst_execute(conduit_cpp::c_node(&exec_params));
  if (err != catalyst_status_ok)
  {
    std::cerr << "Failed to execute Catalyst: " << err << std::endl;
  }
}

// Although no arguments are passed for catalyst_finalize  it is required in
// order to release any resources the ParaViewCatalyst implementation has
// allocated.
void Finalize()
{
  conduit_cpp::Node node;
  catalyst_status err = catalyst_finalize(conduit_cpp::c_node(&node));
  if (err != catalyst_status_ok)
  {
    std::cerr << "Failed to finalize Catalyst: " << err << std::endl;
  }
}
}

#endif
