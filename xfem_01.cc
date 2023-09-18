// ---------------------------------------------------------------------
//
// Copyright (C) 2023 by the deal.II authors
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
 * Dummy assembly of coupling term of
 *
 *   [[u]] = s on gamma
 *
 * with [[u]] = u1-u0. The resulting weak form on gamma:
 *
 *  ([[v]], [[u]]) = (v1-v0, u1-u0)
 *                 = (v0, u0) + (v1, u1) - (v0, u1) - (v1, u0).
 */

#include <deal.II/base/function_signed_distance.h>

#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>

#include <deal.II/non_matching/fe_immersed_values.h>
#include <deal.II/non_matching/fe_values.h>
#include <deal.II/non_matching/mesh_classifier.h>

#include <deal.II/numerics/vector_tools.h>

#include <fstream>


using namespace dealii;

enum ActiveFEIndex
{
  inside      = 0,
  intersected = 1,
  outside     = 1
};

template <int dim>
void
print_sparsity_pattern(const DoFHandler<dim> &dof_handler,
                       const std::string     &file_name)
{
  DynamicSparsityPattern dynamic_sparsity_pattern(dof_handler.n_dofs(),
                                                  dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dynamic_sparsity_pattern);
  SparsityPattern sparsity_pattern;
  sparsity_pattern.copy_from(dynamic_sparsity_pattern);
  std::ofstream out(file_name);
  sparsity_pattern.print_svg(out);
}


template <int dim>
void
test()
{
  // create mesh
  const unsigned int fe_degree = 1;

  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria, 0.0, 1.0);
  tria.refine_global(4);

  // create level-set field and mesh classifier
  DoFHandler<dim> ls_dof_handler(tria);
  ls_dof_handler.distribute_dofs(FE_Q<dim>(fe_degree));

  Vector<double> ls_vector(ls_dof_handler.n_dofs());
  const Functions::SignedDistance::Sphere<dim> signed_distance_sphere;
  VectorTools::interpolate(ls_dof_handler, signed_distance_sphere, ls_vector);

  NonMatching::MeshClassifier<dim> mesh_classifier(ls_dof_handler, ls_vector);
  mesh_classifier.reclassify();

  // create system
  FE_Q<dim>             fe_q(fe_degree);
  FE_Nothing<dim>       fe_n;
  hp::FECollection<dim> fe_collection;
  fe_collection.push_back(FESystem<dim>(fe_q, 1, fe_n, 1));
  fe_collection.push_back(FESystem<dim>(fe_q, 1, fe_q, 1));
  fe_collection.push_back(FESystem<dim>(fe_n, 1, fe_q, 1));

  DoFHandler<dim> dof_handler(tria);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      const auto cell_location = mesh_classifier.location_to_level_set(cell);

      if (cell_location == NonMatching::LocationToLevelSet::inside)
        cell->set_active_fe_index(ActiveFEIndex::inside);
      else if (cell_location == NonMatching::LocationToLevelSet::intersected)
        cell->set_active_fe_index(ActiveFEIndex::intersected);
      else if (cell_location == NonMatching::LocationToLevelSet::outside)
        cell->set_active_fe_index(ActiveFEIndex::outside);
    }

  dof_handler.distribute_dofs(fe_collection);
  print_sparsity_pattern(dof_handler, "sp_0.svg");

  // renumber DoFs to get a block structure
  DoFRenumbering::component_wise(dof_handler);
  print_sparsity_pattern(dof_handler, "sp_1.svg");

  // initialize sparse matrix
  DynamicSparsityPattern dynamic_sparsity_pattern(dof_handler.n_dofs(),
                                                  dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dynamic_sparsity_pattern);
  SparsityPattern sparsity_pattern;
  sparsity_pattern.copy_from(dynamic_sparsity_pattern);
  SparseMatrix<double> stiffness_matrix;
  stiffness_matrix.reinit(sparsity_pattern);

  // assembly sparse matrix
  const QGauss<1> quadrature_1D(fe_degree + 1);

  NonMatching::RegionUpdateFlags region_update_flags;
  region_update_flags.surface = update_values | update_gradients |
                                update_JxW_values | update_quadrature_points |
                                update_normal_vectors;

  NonMatching::FEValues<dim> non_matching_fe_values(fe_collection,
                                                    quadrature_1D,
                                                    region_update_flags,
                                                    mesh_classifier,
                                                    ls_dof_handler,
                                                    ls_vector);

  AffineConstraints<double> dummy;

  std::vector<types::global_dof_index> indices;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      const auto cell_location = mesh_classifier.location_to_level_set(cell);

      non_matching_fe_values.reinit(cell);

      if (cell_location == NonMatching::LocationToLevelSet::inside)
        {
          // nothing to do
        }
      else if (cell_location == NonMatching::LocationToLevelSet::intersected)
        {
          // compute coupling term
          const auto &fe_values =
            *non_matching_fe_values.get_surface_fe_values();

          const unsigned int dofs_per_cell = fe_values.dofs_per_cell;
          FullMatrix<double> local_stiffness(dofs_per_cell, dofs_per_cell);
          indices.resize(dofs_per_cell);
          cell->get_dof_indices(indices);

          FEValuesExtractors::Scalar u_0(0);
          FEValuesExtractors::Scalar u_1(1);

          const auto &fe = cell->get_fe();

          for (const unsigned int i : fe_values.dof_indices())
            for (const unsigned int j : fe_values.dof_indices())
              for (const unsigned int q : fe_values.quadrature_point_indices())
                {
                  const auto i_comp = fe.system_to_component_index(i).first;
                  const auto j_comp = fe.system_to_component_index(j).first;

                  if (i_comp == 0 && j_comp == 0)
                    local_stiffness[i][j] += fe_values.JxW(q) *
                                             fe_values[u_0].value(i, q) *
                                             fe_values[u_0].value(j, q);
                  else if (i_comp == 0 && j_comp == 0)
                    local_stiffness[i][j] += fe_values.JxW(q) *
                                             fe_values[u_1].value(i, q) *
                                             fe_values[u_1].value(j, q);
                  else if (i_comp == 0 && j_comp == 1)
                    local_stiffness[i][j] -= fe_values.JxW(q) *
                                             fe_values[u_0].value(i, q) *
                                             fe_values[u_1].value(j, q);
                  else if (i_comp == 1 && j_comp == 0)
                    local_stiffness[i][j] -= fe_values.JxW(q) *
                                             fe_values[u_1].value(i, q) *
                                             fe_values[u_0].value(j, q);
                }

          dummy.distribute_local_to_global(local_stiffness,
                                           indices,
                                           stiffness_matrix);
        }
      else if (cell_location == NonMatching::LocationToLevelSet::outside)
        {
          // nothing to do
        }
    }
}


int
main()
{
  test<2>();
}