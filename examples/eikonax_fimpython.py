def generate_mesh(num_points):
    mesh_bounds_x = (0, 1)
    mesh_bounds_y = (0, 1)
    num_points_x = num_points
    num_points_y = num_points
    mesh_points_x = np.linspace(*mesh_bounds_x, num_points_x)
    mesh_points_y = np.linspace(*mesh_bounds_y, num_points_y)
    mesh_points = np.column_stack(
        (np.repeat(mesh_points_x, num_points_x), np.tile(mesh_points_y, num_points_y))
    )
    triangulation = Delaunay(mesh_points)
    vertices = triangulation.points
    simplices = triangulation.simplices
    return vertices, simplices