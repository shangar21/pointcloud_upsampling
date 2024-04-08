import bolt
import argparse
import numpy as np
import open3d as o3d
import subprocess
from bolt.visualize import visualize_o3d_pointcloud_o3d, visualize_numpy_pointcloud_o3d
from bolt.distances import chamfer_distance, hausdorff_distance

def mls_sampling(point_cloud_sample_path, k, polynomial_order, output_path, visualize=False, n_final_sample=1024):
    subprocess.run(["pcl_src/build/mls", point_cloud_sample_path, str(k), str(polynomial_order), output_path])
    point_cloud_sample = o3d.io.read_point_cloud(output_path)
    if visualize:
        visualize_o3d_pointcloud_o3d(point_cloud_sample)
    point_cloud_sample = np.asarray(point_cloud_sample.points)
    point_cloud_sample = point_cloud_sample[np.random.choice(point_cloud_sample.shape[0], min(point_cloud_sample.shape[0], n_final_sample), replace=False), :]
    if visualize:
        visualize_numpy_pointcloud_o3d(point_cloud_sample)
    print("MLS shape: ", point_cloud_sample.shape)
    return point_cloud_sample 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--point_cloud_source", type=str, help="Path to the point cloud")
    parser.add_argument("-n", "--n_initial_sample", type=int, help="Number of initial samples", default=1024)
    parser.add_argument("-k", "--k_nearest_neighbours", type=int, help="Number of nearest neighbours", default=30)
    parser.add_argument("-d", "--sigma_d", type=float, help="Sigma_d", default=0.1)
    parser.add_argument("-r", "--sigma_n", type=float, help="Sigma_n", default=0.1)
    parser.add_argument("-i", "--n_iter", type=int, help="Number of iterations", default=1)
    parser.add_argument("-N", "--n_final_sample", type=int, help="Number of final samples", default=2048)
    parser.add_argument("-v", "--visualize", help="Visualize the point cloud", action="store_true")
    parser.add_argument("-t", "--smoothing_type", type=str, help="Type of smoothing", default="bilateral")
    parser.add_argument("-m", "--metric", type=str, help="Type of metric", default="chamfer")
    parser.add_argument("--random_sampling", help="Random sampling", action="store_true")
    parser.add_argument("--mls", help="Use mls sampling", action="store_true")
    parser.add_argument("-o", "--output_path", type=str, help="Output path", default="./bolt_upsampled.npy")
    args = parser.parse_args()
    
    kwargs = vars(args)
    source_path = kwargs.pop('point_cloud_source')
    point_cloud_sample = bolt.upsample.bolt(source_path, **kwargs) 
    np.save(args.output_path, point_cloud_sample)
   
    if args.mls:
        point_cloud = np.load(source_path)
        o3d.io.write_point_cloud("tmp/pointcloud.pcd", o3d.geometry.PointCloud(o3d.utility.Vector3dVector(point_cloud)))
        point_cloud = mls_sampling("tmp/pointcloud.pcd", 20, 2, "tmp/mls.pcd", visualize=args.visualize, n_final_sample=args.n_final_sample)
        np.save(args.output_path, point_cloud)
