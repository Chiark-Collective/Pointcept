import vtk
from vtk.util.numpy_support import vtk_to_numpy
import logging
import laspy


class MeshSampler:
    """
    A class for sampling point clouds from mesh files and saving them in various formats.
    
    Methods:
        generate_cloud(resolution): Generates a point cloud from the mesh at a specified resolution.
        save(output_path): Saves the generated point cloud to a specified path.
    """
    
    def __init__(self, mesh_path, gt_value):
        """
        Initializes the MeshSampler object with the specified mesh file and ground truth value.

        Args:
            mesh_path (str): Path to the mesh file.
            gt_value (float): Ground truth value to assign to each point.
        
        Raises:
            FileNotFoundError: If the mesh_path does not exist or is not a file.
        """
        self.mesh_path = Path(mesh_path)
        if not self.mesh_path.exists() or not self.mesh_path.is_file():
            raise FileNotFoundError(f"Provided mesh path does not exist: {mesh_path}")
        if not isinstance(gt_value, int):
            raise TypeError(f"gt_value must be an integer, got {type(gt_value).__name__} instead.")

        self._points = None
        self._colors = None
        self._normals = None
        self._gt_value = gt_value
    
    @staticmethod
    def check_output_path_viability(output_path):
        """
        Static method to check the viability of an output path.

        Args:
            output_path (str or Path): The output path to validate.

        Raises:
            ValueError: If the output path is None or empty.
            FileNotFoundError: If the directory does not exist and cannot be created.
        """
        if not output_path:
            raise ValueError("Output path is not provided or is empty.")

        # Convert to Path object if necessary
        path = Path(output_path) if not isinstance(output_path, Path) else output_path

        # Check if the directory exists or try to create it
        if not path.parent.exists():
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                logger.info(f"Directory created: {path.parent}")
            except Exception as e:
                logger.error(f"Failed to create directory: {path.parent}")
                raise FileNotFoundError(f"Failed to create directory at {path.parent}: {e}")

        # Optionally, check if path is a valid file path (if you need to ensure it's writable, etc.)
        # This might include checking file permissions or other conditions specific to your environment
        logging.info(f"Output path '{path}' is viable for use.")
    
    def generate_cloud(self, resolution=0.05):
        """
        Generates a sampled point cloud from the mesh at the given resolution.

        Args:
            resolution (float): The distance between sampled points.
        """
        # Step 1: Setup the reader and load the mesh
        reader = vtk.vtkPLYReader()
        reader.SetFileName(self.mesh_path.as_posix())
        reader.Update()
        input_mesh = reader.GetOutput()
        
        # Step 2: Setup Poisson Disk Sampler to sample the mesh
        sampler = vtk.vtkPolyDataPointSampler()
        sampler.SetInputData(input_mesh)
        sampler.SetDistance(resolution)
        sampler.Update()
        # Output of sampler is the point cloud without color and normal data
        sampled_point_cloud = sampler.GetOutput()
        
        # Step 3: Use vtkProbeFilter to interpolate colors onto the sampled points
        probe_filter = vtk.vtkProbeFilter()
        probe_filter.SetInputData(sampled_point_cloud)
        probe_filter.SetSourceData(input_mesh)
        probe_filter.Update()
        
        # The output with interpolated colors
        interpolated_point_cloud = probe_filter.GetOutput()
        
        # Step 4: Compute normals for the sampled points using vtkPolyDataNormals
        normals_filter = vtk.vtkPolyDataNormals()
        normals_filter.SetInputData(interpolated_point_cloud)
        normals_filter.ComputePointNormalsOn()
        normals_filter.Update()
        
        # Get the point data with normals
        normals_point_cloud = normals_filter.GetOutput()
        
        # Convert the interpolated colors to numpy arrays
        self._points = vtk_to_numpy(normals_point_cloud.GetPoints().GetData())
        self._colors = vtk_to_numpy(normals_point_cloud.GetPointData().GetScalars())
        self._normals = vtk_to_numpy(normals_point_cloud.GetPointData().GetNormals())


    def save(self, output_path=None):
        """
        Saves the generated point cloud to a specified file path. The format is determined by the file extension.

        Args:
            output_path (str): Path where the point cloud will be saved.

        Raises:
            ValueError: If an unsupported file extension is provided.
        """
        p = Path(output_path)
        self.check_output_path_viability(p)      
        file_extension = p.suffix
        if file_extension == '.las':
            self._save_as_las(p)
        elif file_extension == '.pth':
            self._save_as_pth()
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}. Please use '.las' or '.pth'.")

    def _save_as_las(self, output_path):
        """
        Saves the point cloud data in LAS format.

        Args:
            output_path (Path): Path to save the LAS file.
        """
        points = self._points
        normals = self._normals
        logger.info("Saving cloud as .las format...")
        header = laspy.LasHeader(version="1.4", point_format=8)
        header.x_scale = 0.001  # Adjust the scale to match the units of your points
        header.y_scale = 0.001
        header.z_scale = 0.001
        
        # Set the offset to the minimum values to avoid loss of precision
        header.offsets = [np.min(points[:, 0]), np.min(points[:, 1]), np.min(points[:, 2])]
        las = laspy.LasData(header)
        # Rescale the colors to .las format
        las_colors = np.round(self._colors * (65535/255)).astype(np.uint16)
        
        # Fill in the point records
        las.x = points[:, 0]
        las.y = points[:, 1]
        las.z = points[:, 2]
        las.red = las_colors[:, 0]
        las.green = las_colors[:, 1]
        las.blue = las_colors[:, 2]
        
        # Add normals as extra dimensions to the LAS file
        las.add_extra_dim(laspy.ExtraBytesParams(name="NormalX", type=np.float32, description="X component of normal vector"))
        las.add_extra_dim(laspy.ExtraBytesParams(name="NormalY", type=np.float32, description="Y component of normal vector"))
        las.add_extra_dim(laspy.ExtraBytesParams(name="NormalZ", type=np.float32, description="Z component of normal vector"))        
        las['NormalX'] = normals[:,0]
        las['NormalY'] = normals[:,1]
        las['NormalZ'] = normals[:,2]

        # Add the ground truth as a scalar field
        las.add_extra_dim(laspy.ExtraBytesParams(name="gt", type=np.float32, description="Ground truth field"))
        las['gt'] = np.broadcast_to(self.gt_value, las.header.point_count)

        las.write(output_path.as_posix())
        logger.info(f"Saved as: {output_path}")
    
    def _save_as_pth(self, output_path):
        """
        Saves the point cloud data in PTH format.

        Args:
            output_path (Path): Path to save the PTH file.
        """
        pass
