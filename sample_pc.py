import bolt
import argparse

'''
todo:
- add chamfer distance
- add hausdorff distances
- CHANGE BILATERAL FILTERING TO ONLY SMOOTH THE POINTS THAT ARE NOT IN THE ORIGINAL POINT CLOUD!!!

'''
def mls_sampling(point_cloud_sample_path, k, polynomial_order, output_path, visualize=False):
    subprocess.run(["pcl_src/build/mls", point_cloud_sample_path, str(k), str(polynomial_order), output_path])
    point_cloud_sample = o3d.io.read_point_cloud(output_path)
    if visualize:
        visualize_o3d_pointcloud_o3d(point_cloud_sample)
    point_cloud_sample = np.asarray(point_cloud_sample.points)
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
    args = parser.parse_args()
    
    kwargs = vars(args)
    point_cloud_sample = bolt.upsample.bolt(kwargs.pop('point_cloud_source'), **kwargs) 
   
  #  mls_sampling("tmp/pointcloud.pcd", 20, 2, "tmp/mls.pcd", visualize=args.visualize)

