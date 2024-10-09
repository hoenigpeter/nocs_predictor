import trimesh
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import time

# Step 1: Load a mesh
mesh = trimesh.load('models/obj_000001.ply')  # Replace with your mesh file path
print(f'Mesh has {len(mesh.vertices)} vertices and {len(mesh.faces)} faces.')

# Step 2: Plot the mesh
mesh.show()  # Display the mesh using trimesh's inbuilt viewer

# Step 3: Convert the mesh to a point cloud (sample points from the surface)
# We'll use open3d for point cloud manipulation
vertices = np.asarray(mesh.vertices)
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(vertices)

# Step 4: Plot the point cloud
o3d.visualization.draw_geometries([point_cloud], window_name='Point Cloud')

# Step 5: Save the point cloud to a file (PLY format)
pointcloud_file = 'pointcloud.ply'
o3d.io.write_point_cloud(pointcloud_file, point_cloud)
print(f'Point cloud saved to {pointcloud_file}')

# Step 6: Reload the point cloud and time the loading process
start_time = time.time()
loaded_point_cloud = o3d.io.read_point_cloud(pointcloud_file)
load_duration = time.time() - start_time

# Step 7: Print loading time and display the reloaded point cloud
print(f'Loading the point cloud took {load_duration:.4f} seconds.')
o3d.visualization.draw_geometries([loaded_point_cloud], window_name='Reloaded Point Cloud')

