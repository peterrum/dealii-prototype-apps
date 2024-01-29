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

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/la_vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>

#include <deal.II/numerics/vector_tools.h>

#include <fstream>

using namespace dealii;

int
main()
{
  const unsigned int dim = 2, degree = 1, n_global_refinements = 1;

  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria, 0, 1, true);
  tria.refine_global(n_global_refinements);

  FE_Q<dim>      fe(degree);
  QGauss<dim>    quad(degree + 1);
  MappingQ1<dim> mapping;

  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  // deal with boundary conditions
  AffineConstraints<double> constraints;
  VectorTools::interpolate_boundary_values(mapping,
                                           dof_handler,
                                           0,
                                           Functions::ConstantFunction<dim>(
                                             1.0),
                                           constraints);
  constraints.close();

  // initialize vectors and system matrix
  Vector<double>       b(dof_handler.n_dofs());
  SparseMatrix<double> A;
  SparsityPattern      sparsity_pattern;

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  sparsity_pattern.copy_from(dsp);
  A.reinit(sparsity_pattern);

  // assemble right-hand side and system matrix
  FullMatrix<double>                   cell_matrix;
  Vector<double>                       cell_rhs;
  std::vector<types::global_dof_index> local_dof_indices;

  FEValues<dim> fe_values(mapping,
                          fe,
                          quad,
                          update_values | update_gradients | update_JxW_values);

  // loop over all cells
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);

      const unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
      cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
      cell_rhs.reinit(dofs_per_cell);

      // loop over cell dofs
      for (const auto q : fe_values.quadrature_point_indices())
        {
          for (const auto i : fe_values.dof_indices())
            for (const auto j : fe_values.dof_indices())
              cell_matrix(i, j) +=
                (fe_values.shape_grad(i, q) * // grad phi_i(x_q)
                 fe_values.shape_grad(j, q) * // grad phi_j(x_q)
                 fe_values.JxW(q));           // dx

          for (const unsigned int i : fe_values.dof_indices())
            cell_rhs(i) += (fe_values.shape_value(i, q) * // phi_i(x_q)
                            1. *                          // f(x_q)
                            fe_values.JxW(q));            // dx
        }

      local_dof_indices.resize(cell->get_fe().dofs_per_cell);
      cell->get_dof_indices(local_dof_indices);

      constraints.distribute_local_to_global(cell_matrix, local_dof_indices, A);

      // constraints.distribute_local_to_global(
      //  cell_matrix, cell_rhs, local_dof_indices, A, b);
    }

  A.print_formatted(std::cout);
}
