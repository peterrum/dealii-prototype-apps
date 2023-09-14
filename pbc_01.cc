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
 * A simple program that presents periodic-boundary conditions and displacement
 * of specified points.
 */

#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>

using namespace dealii;

template <int dim>
SymmetricTensor<4, dim>
get_stress_strain_tensor(const double lambda, const double mu)
{
  SymmetricTensor<4, dim> tmp;
  for (unsigned int i = 0; i < dim; ++i)
    for (unsigned int j = 0; j < dim; ++j)
      for (unsigned int k = 0; k < dim; ++k)
        for (unsigned int l = 0; l < dim; ++l)
          tmp[i][j][k][l] = (((i == k) && (j == l) ? mu : 0.0) +
                             ((i == l) && (j == k) ? mu : 0.0) +
                             ((i == j) && (k == l) ? lambda : 0.0));
  return tmp;
}

template <int dim>
inline SymmetricTensor<2, dim>
get_strain(const FEValues<dim> &fe_values,
           const unsigned int   shape_func,
           const unsigned int   q_point)
{
  SymmetricTensor<2, dim> tmp;

  for (unsigned int i = 0; i < dim; ++i)
    tmp[i][i] = fe_values.shape_grad_component(shape_func, q_point, i)[i];

  for (unsigned int i = 0; i < dim; ++i)
    for (unsigned int j = i + 1; j < dim; ++j)
      tmp[i][j] = (fe_values.shape_grad_component(shape_func, q_point, i)[j] +
                   fe_values.shape_grad_component(shape_func, q_point, j)[i]) /
                  2;

  return tmp;
}


/**
 * A helper function that applies a displacement at a given point and 
 * modifies the constraints appropriately. 
 */
template <int dim, int spacedim, typename Number>
void
apply_displacement_at_point(const Mapping<dim, spacedim> &   mapping,
                            const DoFHandler<dim, spacedim> &dof_handler,
                            const Point<spacedim> &          displacement,
                            const Point<spacedim> &          point,
                            const ComponentMask &            component_mask,
                            AffineConstraints<Number> &      constraints)
{
  const auto &fe = dof_handler.get_fe();

  FEValues<dim> fe_values(mapping,
                          fe,
                          Quadrature<dim>(fe.get_generalized_support_points()),
                          update_quadrature_points);

  std::vector<types::global_dof_index> local_dof_indices;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (cell->is_locally_owned() == false)
        continue;

      fe_values.reinit(cell);

      const auto &points = fe_values.get_quadrature_points();

      unsigned int c = numbers::invalid_unsigned_int;

      for (unsigned int i = 0; i < points.size(); ++i)
        {
          if (point.distance(points[i]) < 1e-10)
            c = i;
        }

      if (c == numbers::invalid_unsigned_int)
        continue;

      local_dof_indices.resize(cell->get_fe().dofs_per_cell);
      cell->get_dof_indices(local_dof_indices);

      for (unsigned int comp = 0; comp < dim; ++comp)
        if (component_mask[comp])
          {
            const unsigned int index =
              local_dof_indices[fe.component_to_system_index(comp, c)];
            constraints.add_line(index);
            constraints.set_inhomogeneity(index, displacement[comp]);
          }

      return; // it is enough to add the constraints once
    }

  AssertThrow(false, ExcInternalError()); // the point could not be found
}


int
main()
{
  const unsigned int dim = 2, degree = 1, n_refinements = 2;

  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria, 0.0, 1.0, true);
  tria.refine_global(n_refinements);

  // collect periodic faces
  std::vector<
    GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
    periodic_faces;
  GridTools::collect_periodic_faces(tria, 0, 1, 0, periodic_faces);
  GridTools::collect_periodic_faces(tria, 2, 3, 1, periodic_faces);
  tria.add_periodicity(periodic_faces);

  FESystem<dim>        fe(FE_Q<dim>(degree), dim);
  QGauss<dim>          quad(degree + 1);
  MappingQGeneric<dim> mapping(1);

  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  // Create constraint matrix
  AffineConstraints<double> constraints;

  // ... constrain points
  apply_displacement_at_point(mapping,
                              dof_handler,
                              {0.0, 0.0},
                              {0.0, 0.0},
                              std::vector<bool>{true, true},
                              constraints);
  apply_displacement_at_point(mapping,
                              dof_handler,
                              {0.1, 0.0},
                              {0.0, 0.5},
                              std::vector<bool>{true, false},
                              constraints);

  // ... apply pbc
  DoFTools::make_periodicity_constraints(dof_handler, 0, 1, 0, constraints);
  DoFTools::make_periodicity_constraints(dof_handler, 2, 3, 1, constraints);

  // ... finalize constraints
  constraints.close();

  const auto stress_strain_tensor_1 =
    get_stress_strain_tensor<dim>(9.695e10, 7.617e10);

  const auto stress_strain_tensor_2 =
    get_stress_strain_tensor<dim>(10 * 9.695e10, 7.617e10);

  for (const auto &cell : tria.active_cell_iterators())
    cell->set_material_id(cell->center()[0] > 0.5);

  Vector<double>       x(dof_handler.n_dofs()), b(dof_handler.n_dofs());
  SparseMatrix<double> A;
  SparsityPattern      sparsity_pattern;

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints);
  sparsity_pattern.copy_from(dsp);
  A.reinit(sparsity_pattern);

  FEValues<dim> fe_values(mapping,
                          fe,
                          quad,
                          update_gradients | update_JxW_values);

  FullMatrix<double>                   cell_matrix;
  Vector<double>                       cell_rhs;
  std::vector<types::global_dof_index> local_dof_indices;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (cell->is_locally_owned() == false)
        continue;

      fe_values.reinit(cell);

      const unsigned int dofs_per_cell = cell->get_fe().dofs_per_cell;
      cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
      cell_rhs.reinit(dofs_per_cell);

      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        for (unsigned int j = 0; j < dofs_per_cell; ++j)
          for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
            {
              const auto eps_phi_i = get_strain(fe_values, i, q);
              const auto eps_phi_j = get_strain(fe_values, j, q);

              cell_matrix(i, j) +=
                (eps_phi_i *
                 (cell->material_id() == 1 ? stress_strain_tensor_1 :
                                             stress_strain_tensor_2) *
                 eps_phi_j) *
                fe_values.JxW(q);
            }


      local_dof_indices.resize(cell->get_fe().dofs_per_cell);
      cell->get_dof_indices(local_dof_indices);

      constraints.distribute_local_to_global(
        cell_matrix, cell_rhs, local_dof_indices, A, b);
    }

  ReductionControl         reduction_control;
  SolverCG<Vector<double>> solver(reduction_control);
  solver.solve(A, x, b, PreconditionIdentity());

  printf("Solved in %d iterations.\n", reduction_control.last_step());

  constraints.distribute(x);

  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  x.update_ghost_values();
  data_out.add_data_vector(
    dof_handler,
    x,
    "solution",
    std::vector<DataComponentInterpretation::DataComponentInterpretation>(
      dim, DataComponentInterpretation::component_is_part_of_vector));
  data_out.build_patches(mapping, degree);

  std::ofstream output("solution.vtu");
  data_out.write_vtu(output);
}
