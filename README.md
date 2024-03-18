# Implementation of Point Cloud Upsampling with Bilateral Filter and Octree

## Setup

Before starting install, need some dependencies:

```
pip install "pybind[global]"
sudo apt install libpcl-dev
sudo apt install libeigen3-dev
sudo apt install libnanoflann-dev
```

### BOLT ⚡

Make sure environment is up to date with requirements.txt

```bash
pip install -r requirements.txt
```

If cloned, can also install and set-up with:

```bash
pip install -e .
```

To avoid cloning, can alternatively pip install with the following, however you will not be able to run MLS point upsampling

```bash
pip install git+https://github.com/shangar21/pointcloud_upsampling.git                                                                            
```

### MLS resampling

To build the executable to use MLS resampling perform the following:

1. First install point cloud library (pcl), with ubuntu/debian its quite simple (see below), for other platforms refer to the documentation [here](https://pointclouds.org/)

```bash
sudo apt install libpcl-dev
```

2. After cloning and changing directories to the directory, compile the program

```bash
cd pcl_src && mkdir build && cd build
cmake ../
cmake --build .
```
After its built, you should be able to run and reconstruct surfaces with MLS with:

```
<path to git repo>/pcl_src/build/mls <path to pcd file> <number of neighbours> <polynomial degree> <path to output pcd>
```
An example:

```bash
pcl_src/build/mls ./tmp/pointcloud.pcd 30 2 ./tmp/pointcloud_up.pcd
```
the above call runs MLS surface reconstruction with 30 neighbours, and a polynomial of degree 2 of the point cloud in `pointcloud.pcd` and outputs an upsampled pointcloud to `pointcloud_up.pcd`

## Dataset

The dataset used to evaluate the model is found [here](https://github.com/stevenygd/PointFlow#dataset), it is derived from the ShapeNet dataset.

## Running

### BOLT ⚡

Since BOLT can be used as a standalone upsampling library, a sample usage case can be found in `sample_pc.py` in the root directory. The only parameter needed to run BOLT is a path to an npy file. An example below:

```python
import bolt

point_cloud_path = "/media/storage/ShapeNetCore.v2.PC15k/02691156/train/1021a0914a7207aff927ed529ad90a11.npy"

upsampled_point = bolt.upsample.bolt(point_cloud_path)
```

We describe a list of keyword arguments/hyperparameters that may help with better results:

| Argument    | Description |
| -------- | ------- |
| `n_initial_sample`  | The number of points to initially sample from the ground truth / source point cloud, default is 1024    |
| `k_nearest_neighbours` | The number of neighbours used in bilateral filtering as well ass KNN smoothing, default is 30     |
| `sigma_d`    | The sigma_d parameter used in one of the weight gaussians (refer to paper for more specifics)    |
| `sigma_n`    | The sigma_n parameter used in one of the weight gaussians (refer to paper for more specifics)    |
| `n_iter`     | Number of times to run the bilateral filtering method on the new points inserted by octree, default is 1 | 
| `n_final_sample` | number of desired points in the final upsampled point cloud, default is 2048 (must be a multiple of `n_initial_sample`) |
| `visualize` | a boolean to show a visualization of each stage (initial point cloud, coarse octree upsampling, final smoothed point cloud) |
| `smoothing_type` | a string either `knn` or `bilateral`, determines what smoothing algorithm is used after upsampling with octree | 
| `random_sample` | a boolean to skip using an octree and plot points randomly instead, mainly used to compare octree vs no octree performance | 

The BOLT library also has a bunch of seperate useful tools and classes, such as an Octree implementation, as well as the actual implementation of both the KNN and Bilateral smoothing methods. 

### Octree

The tree is a very simple implementation, there is no depth parameter. Simply create one with the following:

```python
import bolt

# Assume point_cloud is a numpy array of shape (N, 3) 
octree = bolt.octree.Octree()
octree.create(point_cloud)
```
This will read each point one by one and insert. Other functions that may be useful are `deepest_parent` which for a point find the direct parent, `get_points` which gets all points stored in the tree.

### Smoothing

The smoothing module contains the smoothing methods used in BOLT, KNN and Bilateral. Can be called as below:

```python
import bolt

# Assume point_cloud is the original sample of shape (N, 3) and new_points are the points we wish to add to our point_cloud, also with a shape (N', 3)
point_cloud_up = bolt.smoothing.nearest_neighbours_smoothing(point_cloud, new_points, k=30) # k is the number of neighbours
point_cloud_up = bolt.smoothing.bilateral_smoothing(point_cloud, new_points, k=30, sigma_d=0.1, sigma_n=0.1, n_iter=1)
```
