/*
 * SPDX-FileCopyrightText: 2026 Oak Ridge National Laboratory and Contributors
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __WRITER_H__
#define __WRITER_H__

#include <adios2.h>
#include <mpi.h>
#include "lulesh.h"
class Writer
{
public:
    Writer(Domain &domain, adios2::IO io);
    void open(const std::string &fname, bool append);
    void write(int step, Domain &domain);
    void close();

protected:
    adios2::IO io;
    adios2::Engine writer;
    adios2::Variable<int> var_connec; 
    adios2::Variable<float> var_coords;
    adios2::Variable<double> var_e;
    adios2::Variable<double> var_p;
    adios2::Variable<double> var_q;
    adios2::Variable<double> var_v;
    adios2::Variable<double> var_ss;
    adios2::Variable<double> var_eleMass;
    adios2::Variable<double> var_nodalMass;
    adios2::Variable<double> var_velocity;
    adios2::Variable<double> var_acceleration;
    adios2::Variable<double> var_force;
    adios2::Variable<int> var_step;
    int internal_step; 
};

#endif
