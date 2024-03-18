#include<iostream>
#include<pcl/io/pcd_io.h>
#include<pcl/point_types.h>
#include<pcl/surface/mls.h>
#include<pcl/search/kdtree.h>
#include<string>
#include<fstream>
#include<chrono>

void handleOutputFile(std::string output_file)
{
	std::ifstream file(output_file);
	if(!file.good())
	{
		std::ofstream outfile(output_file);
	}
}


float estimateSearchRadius(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, int k)
{
	float radius = 0.0;
	pcl::search::KdTree<pcl::PointXYZ>::Ptr kdTree(new pcl::search::KdTree<pcl::PointXYZ>);
	std::vector<int> pointIdxNKNSearch(k);
	std::vector<float> pointNKNSquaredDistance(k);

	kdTree->setInputCloud(cloud);
  
	for(size_t i = 0; i < cloud->points.size(); i++){
		if(kdTree->nearestKSearch(cloud->points[i], k, pointIdxNKNSearch, pointNKNSquaredDistance) > 0){
			radius += sqrt(pointNKNSquaredDistance[k-1]);
		}
	}

	return radius/cloud->points.size();
}

int main(int argc, char** argv){

	std::string input_file = std::string(argv[1]);
	int k = std::stoi(argv[2]);
	int polynomial_order = std::stoi(argv[3]);
	std::string output_file = std::string(argv[4]);
	std::cout << "input file: " << input_file << std::endl;

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr outputCloud (new pcl::PointCloud<pcl::PointXYZ>);

	if(pcl::io::loadPCDFile<pcl::PointXYZ>(input_file, *cloud) == -1){
		PCL_ERROR("Couldn't read file %s\n", input_file.c_str());
		return -1;
	}

	pcl::search::KdTree<pcl::PointXYZ>::Ptr kdTree(new pcl::search::KdTree<pcl::PointXYZ>);
	pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointXYZ> mls;

	auto start = std::chrono::high_resolution_clock::now();
	
	float radius = estimateSearchRadius(cloud, k);
	mls.setComputeNormals(true);
	mls.setInputCloud(cloud);
	mls.setSearchMethod(kdTree);
	mls.setSearchRadius(radius);
	mls.setUpsamplingMethod(pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointXYZ>::UpsamplingMethod::SAMPLE_LOCAL_PLANE);
	mls.setUpsamplingRadius(radius);
	mls.setUpsamplingStepSize(0.01);
	mls.setPolynomialOrder(polynomial_order);
	mls.setSqrGaussParam(radius*radius);
	mls.process(*outputCloud);

	auto end = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> elapsed = end - start;

	std::cout << "Time taken: " << elapsed.count() << "s\n";

	handleOutputFile(output_file);
	pcl::io::savePCDFile(output_file, *outputCloud);

  return (0);
}

