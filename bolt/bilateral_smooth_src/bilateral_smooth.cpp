#include <numeric>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <nanoflann.hpp>
#include <iostream>

namespace py = pybind11;

Eigen::Vector3f regressionNormal(Eigen::MatrixXf neighnours){
	Eigen::Vector3f mean = neighnours.colwise().mean();
	Eigen::MatrixXf centered = neighnours.rowwise() - mean.transpose();
	Eigen::Matrix3f cov = (centered.adjoint() * centered) / (neighnours.rows() - 1);
	Eigen::EigenSolver<Eigen::Matrix3f> es(cov);
	Eigen::Vector3f normal = es.eigenvectors().col(0).real();
	return normal;
}

Eigen::MatrixXf slice(Eigen::MatrixXf basePoints, std::vector<int> indices){
	Eigen::MatrixXf sliced(indices.size(), basePoints.cols());
	for(int i = 0; i < indices.size(); i++){
		sliced.row(i) = basePoints.row(indices[i]);
	}
	return sliced;
}

void printVector(std::vector<int> vec){
	for(int i = 0; i < vec.size(); i++){
		std::cout << vec[i] << " ";
	}
	std::cout << std::endl;
}

void printEigenMatrix(Eigen::MatrixXf matrix){
	for(int i = 0; i < matrix.rows(); i++){
		for(int j = 0; j < matrix.cols(); j++){
			std::cout << matrix(i, j) << " ";
		}
		std::cout << std::endl;
	}
}

void printEigenVector(Eigen::Vector3f vec){
	for(int i = 0; i < vec.size(); i++){
		std::cout << vec[i] << " ";
	}
	std::cout << std::endl;
}

Eigen::MatrixXf bilateralSmooth(Eigen::MatrixXf basePoints, Eigen::MatrixXf newPoints, int k, float sigmaD, float sigmaN, int nIter){
	std::cout << "Starting bilateral smooth with " << newPoints.rows() << " points" << std::endl;
	nanoflann::KDTreeEigenMatrixAdaptor<Eigen::MatrixXf> kdTree(3, basePoints);
	for(int nil = 0; nil < nIter; nil++){
		for(int i = 0; i < newPoints.rows(); i++){
			Eigen::Vector3f point = newPoints.row(i);
			std::vector<int> indices(k);
			std::vector<float> dists(k);
			nanoflann::KNNResultSet<float, int, int> resultSet(k);
			resultSet.init(&indices[0], &dists[0]);
			kdTree.index->findNeighbors(resultSet, point.data(), nanoflann::SearchParams(k));
			Eigen::MatrixXf neighbours = slice(basePoints, indices);
			Eigen::Vector3f normal = regressionNormal(neighbours);
			float sumW = 0;
			float deltaP = 0;
			for (int j = 0; j < neighbours.rows(); j++){
				Eigen::Vector3f neighbour = neighbours.row(j);
				float dist_d = (point - neighbour).norm();
				float dist_n = (neighbour - point).dot(normal);
				float w_d = exp(-dist_d * dist_d / (2 * sigmaD * sigmaD));
				float w_n = exp(-dist_n * dist_n / (2 * sigmaN * sigmaN));
				float w = w_d * w_n;
				deltaP += w * dist_n;
				sumW += w;
			}
			newPoints.row(i) = point + ((deltaP / sumW) * normal);
		}
	}
	return newPoints;
}

Eigen::MatrixXf pyBilateralSmooth(Eigen::MatrixXf basePoints, Eigen::MatrixXf newPoints, int k, float sigmaD, float sigmaN, int nIter){
	Eigen::MatrixXf result = bilateralSmooth(basePoints, newPoints, k, sigmaD, sigmaN, nIter);
	return result;
}

PYBIND11_MODULE(bilateral_smooth_src, m){
	m.def("bilateral_smooth", &pyBilateralSmooth, "Convert a numpy array to an Eigen matrix");
}
