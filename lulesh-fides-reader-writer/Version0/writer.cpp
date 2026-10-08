
#include "writer.h"
#include <iostream>
#include <iomanip>

Writer::Writer(Domain &domain, adios2::IO io)
: io(io)
{

    // * Connectivity array
    int ci = 0 ;
    int *conn = new int[domain.numElem()*8] ;
    for (int ei=0; ei < domain.numElem(); ++ei) {
      Index_t *elemToNode = domain.nodelist(ei) ;
      for (int ni=0; ni < 8; ++ni) {
         conn[ci++] = elemToNode[ni] ;
      }
    }


   const char* coordnames[3] = {"X", "Y", "Z"};
   float *coords[3] ;
   coords[0] = new float[domain.numNode()] ;
   coords[1] = new float[domain.numNode()] ;
   coords[2] = new float[domain.numNode()] ;
   for (int ni=0; ni < domain.numNode() ; ++ni) {
      coords[0][ni] = float(domain.x(ni)) ;
      coords[1][ni] = float(domain.y(ni)) ;
      coords[2][ni] = float(domain.z(ni)) ;
   }


    // * add attributes for Fides, for unstructured grid
    io.DefineAttribute<std::string>("Fides_Data_Model", "unstructured_single"); // * Type of model
    io.DefineAttribute<std::string>("Fides_Cell_Type", "hexahedron"); // * Type of element, 8 vertices and 6 faces 

    io.DefineAttribute<std::string>("Fides_Coordinates_Variable", "coords"); // * Coordinates
    io.DefineAttribute<std::string>("Fides_Connectivity_Variable", "connectivity"); // * Connectivity to outline elements
    
    // ! num_verts, cell_types need to be defined like "U"


    std::vector<std::string> varList = {"e", "p", "q", "v", "ss", "eleMass", "nodalMass", "velocity", "acceleration", "force"}; // * What variables you intend to pass
    std::vector<std::string> assocList = {"cells", "cells", "cells", "cells", "cells", "cells", "points", "points","points","points"}; // * Cell or point association?

    io.DefineAttribute<std::string>("Fides_Variable_List", varList.data(), varList.size());
    io.DefineAttribute<std::string>("Fides_Variable_Associations", assocList.data(),
                                    assocList.size());

    std::cout << "number of elements: "<< domain.numElem() << std::endl;
    std::cout << "number of nodes: " << domain.numNode() <<std::endl;


   var_connec = io.DefineVariable<int>("connectivity", {domain.numElem()*8}, {0}, {domain.numElem()*8});
      
   var_coords = io.DefineVariable<float>("coords", {domain.numNode(), 3}, {0, 0}, {domain.numNode(), 3});

    var_e = io.DefineVariable<double>("e", {domain.numElem()}, 
                                      {0},
                                      {domain.numElem()});

    var_p = io.DefineVariable<double>("p", {domain.numElem()}, 
                                      {0},
                                      {domain.numElem()});

    var_q = io.DefineVariable<double>("q", {domain.numElem()}, 
                                      {0},
                                      {domain.numElem()});

    var_v = io.DefineVariable<double>("v", {domain.numElem()}, 
                                      {0},
                                      {domain.numElem()});


    var_ss = io.DefineVariable<double>("ss", {domain.numElem()}, 
                                      {0},
                                      {domain.numElem()});

    var_eleMass = io.DefineVariable<double>("eleMass", {domain.numElem()},
                                      {0},
                                      {domain.numElem()});


    var_nodalMass = io.DefineVariable<double>("nodalMass", {domain.numNode()},
                                        {0},
                                      {domain.numNode()});

    var_velocity = io.DefineVariable<double>("velocity", {domain.numNode()},
                                      {0},
                                      {domain.numNode()});

    var_acceleration = io.DefineVariable<double>("acceleration",{domain.numNode()},
                                      {0},
                                      {domain.numNode()});

    var_force = io.DefineVariable<double>("force",{domain.numNode()},
                                      {0},
                                      {domain.numNode()});

    var_step = io.DefineVariable<int>("step", {1}, {0}, {1});
}

void Writer::open(const std::string &fname, bool append)
{
    adios2::Mode mode = adios2::Mode::Write; // * Set it up as a data producer
    if (append)
    {
        mode = adios2::Mode::Append;
    }
    writer = io.Open(fname, mode); // * Setup the "engine", in our case its still bp

    std::string engineType = writer.Type();
    std::cout << "Active Engine: " << engineType << std::endl;

}


void Writer::write(int step, Domain &domain)
{

    // * Connectivity array
    int ci = 0 ;
    //int *conn = new int[domain.numElem()*8] ;
    std::vector<int> conn;
    for (int ei=0; ei < domain.numElem(); ++ei) {
      Index_t *elemToNode = domain.nodelist(ei) ;
      for (int ni=0; ni < 8; ++ni) {
         conn.push_back(elemToNode[ni]) ;
      }
    }


   const char* coordnames[3] = {"X", "Y", "Z"};
   
    // * Must flatten 2D array to 1D for ADIOS, 
    // * variable shape definition should restructure it correctly

    std::vector<float> coords;
   for (int ni=0; ni < domain.numNode() ; ++ni) {
     coords.push_back(float(domain.x(ni)));
     coords.push_back(float(domain.y(ni)));
     coords.push_back(float(domain.z(ni)));
   }

    std::vector<double> e = domain.e_vec();
    std::vector<double> p = domain.p_vec();
    std::vector<double> q = domain.q_vec();
    std::vector<double> v = domain.v_vec();
    std::vector<double> ss = domain.ss_vec();
    std::vector<double> eleMass = domain.eleMass();
    
    std::vector<double> nodalMass = domain.nodalMass(); 

    std::vector<double> velocity_x = domain.xd();
    std::vector<double> velocity_y = domain.yd();
    std::vector<double> velocity_z = domain.zd();
    
    std::vector<double> velocity_mag;

    for (int i =0; i < velocity_x.size(); i++) {
        velocity_mag.push_back(std::sqrt(std::pow(velocity_x[i], 2) + std::pow(velocity_y[i], 2) + std::pow(velocity_z[i],2)));
    }


    std::vector<double> acceleration_x = domain.xdd();
    std::vector<double> acceleration_y = domain.ydd();
    std::vector<double> acceleration_z = domain.zdd();
    
    std::vector<double> acceleration_mag;

    for (int i =0; i < acceleration_x.size(); i++) {
        acceleration_mag.push_back(std::sqrt(std::pow(acceleration_x[i], 2) + std::pow(acceleration_y[i], 2) + std::pow(acceleration_z[i],2)));
    }


    std::vector<double> force_x = domain.fx();
    std::vector<double> force_y = domain.fy();
    std::vector<double> force_z = domain.fz();
    
    std::vector<double> force_mag;

    for (int i =0; i < force_x.size(); i++) {
        force_mag.push_back(std::sqrt(std::pow(force_x[i], 2) + std::pow(force_y[i], 2) + std::pow(force_z[i],2)));
    }

    writer.BeginStep();

    writer.Put<int>(var_connec, conn.data());
    writer.Put<float>(var_coords, coords.data());
    
    
    writer.Put<double>(var_e, e.data());
    writer.Put<double>(var_p, p.data());
    writer.Put<double>(var_q, q.data());
    writer.Put<double>(var_v, v.data());
    writer.Put<double>(var_ss, ss.data());
    writer.Put<double>(var_nodalMass, nodalMass.data());
    writer.Put<double>(var_eleMass, eleMass.data());

    writer.Put<double>(var_velocity, velocity_mag.data());
    writer.Put<double>(var_acceleration, acceleration_mag.data());
    writer.Put<double>(var_force, force_mag.data());

    writer.Put<int>(var_step, &step);

    writer.EndStep();
    internal_step++;
}

void Writer::close() { writer.Close(); }
