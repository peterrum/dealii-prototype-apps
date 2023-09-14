// ---------------------------------------------------------------------
//
// Copyright (C) 2021 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of deal.II.
//
// ---------------------------------------------------------------------

/**
 * A simple program that presents how to apply changing dbcs.
 */

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/block_vector.h>

using namespace dealii;

int
main()
{
  // create dummy system
  const unsigned int dim = 2;

  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria, 0.0, 1.0, true);

  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(FE_Q<dim>(1));

  // setup constraints
  AffineConstraints<double> constraints;
  constraints.add_line(0);
  constraints.add_entry(0, dof_handler.n_dofs(), 1.0);
  constraints.close();

  // creeate dummy vector and apply constraints
  BlockVector<double> vector(
    std::vector<types::global_dof_index>{dof_handler.n_dofs(), 1});
  vector.collect_sizes();
  vector[dof_handler.n_dofs()] = 1;

  vector.print(std::cout);
  std::cout << std::endl;

  constraints.distribute(vector);
  vector.print(std::cout);
  ;
  std::cout << std::endl;

  // modify dbc
  vector[dof_handler.n_dofs()] = 2;

  vector.print(std::cout);
  std::cout << std::endl;

  constraints.distribute(vector);
  vector.print(std::cout);
}