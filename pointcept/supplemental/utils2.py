import vtk


def read_ply_mesh(file_path, compute_normals=True):
    """
    Reads a .ply mesh from a file path and optionally computes normals.

    Args:
        file_path (str): Path to the .ply file.
        compute_normals (bool): If True, computes normals for the mesh. Default is True.

    Returns:
        vtk.vtkPolyData: A VTK object representing the loaded mesh, optionally with normals computed.
    """
    # Read the PLY file
    reader = vtk.vtkPLYReader()
    reader.SetFileName(file_path)
    reader.Update()

    # Get the output as vtkPolyData
    polyData = reader.GetOutput()

    # Optionally compute normals for the mesh
    if compute_normals:
        normals_filter = vtk.vtkPolyDataNormals()
        normals_filter.SetInputData(polyData)
        normals_filter.ComputePointNormalsOn()
        normals_filter.Update()
        return normals_filter.GetOutput()

    # Return the vtkPolyData object without normals
    return polyData

def render_vtk_mesh(mesh, window_name="VTK Mesh Viewer", background_color=(0.1, 0.2, 0.4)):
    """
    Renders a vtkPolyData mesh using VTK in a Jupyter notebook.

    Args:
        mesh (vtk.vtkPolyData): The vtkPolyData mesh to render.
        window_name (str): The title of the render window.
        background_color (tuple): Background color in RGB format.
    """
    # Create a mapper and actor for the mesh
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(mesh)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(1.0, 1.0, 1.0)  # Set mesh color to white for better visibility
    actor.GetProperty().SetEdgeVisibility(1)  # Optional: show edges
    actor.GetProperty().SetLineWidth(1.0)  # Optional: line width for edges

    # Create a renderer and render window
    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(800, 600)  # Set window size
    render_window.SetWindowName(window_name)

    # Set the background color
    renderer.SetBackground(*background_color)

    # Add actor to the renderer
    renderer.AddActor(actor)

    # Create a render window interactor
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    # Start the rendering loop
    render_window.Render()
    render_window_interactor.Start()