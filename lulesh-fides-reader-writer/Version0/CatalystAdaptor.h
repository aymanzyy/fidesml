// SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
// SPDX-License-Identifier: BSD-3-Clause
#ifndef CatalystAdaptor_h
#define CatalystAdaptor_h

#include <catalyst.hpp>
#include <fstream> 
#include <iostream>
#include <string>

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
void Initialize(int argc, char* argv[], std::string JSONFileName, std::string cat_script, std::string address)
{
  conduit_cpp::Node node;

  node["catalyst/scripts/script/filename"].set_string(cat_script);
  node["catalyst_load/implementation"] = "paraview";
  //node["catalyst_load/search_paths/paraview"] = "/home/local/KHQ/ayman.yousef/Downloads/paraview_new_fides_writer/paraview_build/lib/catalyst";
  node["catalyst_load/search_paths/paraview"] = PARAVIEW_IMPL_DIR;

  // * Setup the fides specific parameters for PV

  catalyst_status err = catalyst_initialize(conduit_cpp::c_node(&node));
  if (err != catalyst_status_ok)
  {
    std::cerr << "Failed to initialize Catalyst: " << err << std::endl;
  }
}

void Execute(int cycle, double time, int argc, char* argv[], Domain& localDomain, std::string myRole, int nx)
{


  if (myRole == "writer") {
    conduit_cpp::Node exec_params;  
    auto state = exec_params["catalyst/state"];
    state["timestep"].set(cycle);
    state["time"].set(time);
    state["multiblock"].set(1);

    auto channel = exec_params["catalyst/channels/grid"];

    channel["type"].set("mesh");
    auto mesh = channel["data"];

    std::vector<float> coordsX;
    std::vector<float> coordsY;
    std::vector<float> coordsZ;

    std::vector<float> velocityMag; 
    for (int ni=0; ni < localDomain.numNode() ; ++ni) {
      coordsX.push_back(float(localDomain.x(ni)));
      coordsY.push_back(float(localDomain.y(ni)));
      coordsZ.push_back(float(localDomain.z(ni)));
    }


    int ci = 0 ;
    //int *conn = new int[domain.numElem()*8] ;
    std::vector<int64_t> conn;
    for (int ei=0; ei < localDomain.numElem(); ++ei) {
      Index_t *elemToNode = localDomain.nodelist(ei) ;
      for (int ni=0; ni < 8; ++ni) {
         conn.push_back(elemToNode[ni]) ;
      }
    }


    mesh["coordsets/coordinates/type"] = "explicit";
    
    mesh["coordsets/coordinates/values/x"].set_external(coordsX);
    mesh["coordsets/coordinates/values/y"].set_external(coordsY);
    mesh["coordsets/coordinates/values/z"].set_external(coordsZ);
    
    // * FidesWriter only supports unstructured, so we'll switch this out


    mesh["topologies/mesh/type"] = "unstructured";
    mesh["topologies/mesh/coordset"] =  "coordinates";
    mesh["topologies/mesh/elements/shape"] = "hex";
    mesh["topologies/mesh/elements/connectivity"].set_external(conn);

    
    mesh["fields/velocity/association"] = "vertex";
    mesh["fields/velocity/topology"] = "mesh";

    // * Calculate velocity mag
    for (int ni=0; ni < localDomain.numNode() ; ++ni) {
      float velocity_mag =sqrt((float(localDomain.xd(ni)) * float(localDomain.xd(ni))) + (float(localDomain.yd(ni)) * float(localDomain.yd(ni))) + (float(localDomain.zd(ni)) * float(localDomain.zd(ni))));
      velocityMag.push_back(velocity_mag);
    }

    mesh["fields/velocity/values/"].set_external(velocityMag);
    
    if (cycle == 1) {
      exec_params.print();
    }
    
    catalyst_status err = catalyst_execute(conduit_cpp::c_node(&exec_params));
    
    if (err != catalyst_status_ok)
    {
      std::cerr << "Failed to execute Catalyst: " << err << std::endl;
    } 


  } else if (myRole == "reader"){
    conduit_cpp::Node exec_params;  

    auto state = exec_params["catalyst/state"];
    state["timestep"].set(cycle);
    state["time"].set(time);
    state["multiblock"].set(1);

    exec_params["catalyst/channels/grid/type"].set("mesh");
    exec_params["catalyst/channels/grid/data"].set(localDomain.node());

    catalyst_status err = catalyst_execute(conduit_cpp::c_node(&exec_params));
    if (err != catalyst_status_ok)
    {
      std::cerr << "Failed to execute Catalyst: " << err << std::endl;
    } 
  }
}
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
