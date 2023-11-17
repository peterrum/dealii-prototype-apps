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
 * A simple program that shows ho use boost data structures to find
 * intersections between 1D elements spheres.
 */

#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_cache.h>

using namespace dealii;

template <int dim, int spacedim>
std::vector<typename Triangulation<dim, spacedim>::active_cell_iterator>
find_cells_potentially_intersecting_sphere(
  const GridTools::Cache<dim, spacedim> &cache,
  const Point<spacedim> &                point,
  const double                           radius)
{
  auto p1 = point;
  auto p2 = point;

  for (int d = 0; d < spacedim; ++d)
    {
      p1[d] = p1[d] - radius;
      p2[d] = p2[d] + radius;
    }

  BoundingBox<spacedim> bb({p1, p2});

  std::vector<typename Triangulation<dim, spacedim>::active_cell_iterator>
    result;

  auto cell = cache.get_cell_bounding_boxes_rtree().qbegin(
    boost::geometry::index::intersects(bb));
  const auto end = cache.get_cell_bounding_boxes_rtree().qend();

  for (; cell != end; ++cell)
    result.emplace_back(cell->second);

  return result;
}

int
main()
{
  const unsigned int dim      = 1;
  const unsigned int spacedim = 2;
  const unsigned int n_cells  = 100;

  Triangulation<dim, spacedim> tria;

  std::vector<Point<spacedim>> vertices;

  for (unsigned int i = 0; i < n_cells; ++i)
    {
      const double phi = (1.0 * i) / n_cells * 2.0 * numbers::PI;
      vertices.emplace_back(std::cos(phi), std::sin(phi));
    }

  std::vector<::CellData<dim>> cells(n_cells);

  for (unsigned int i = 0; i < n_cells; ++i)
    {
      cells[i].vertices[0] = i;
      cells[i].vertices[1] = (i + 1) % n_cells;
    }

  tria.create_triangulation(vertices, cells, {});

  GridTools::Cache<dim, spacedim> cache(tria);

  const auto neighboring_list =
    find_cells_potentially_intersecting_sphere<dim, spacedim>(
      cache, Point<spacedim>{1.0, 0.0}, 0.5);

  for (const auto &cell : neighboring_list)
    std::cout << cell->center() << std::endl;
}
