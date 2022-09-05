#include <iostream>
#include <cmath>
#include "compress_weights.h"

CompressWeights::CompressWeights(double user_percent){
    this->percentage=user_percent;
}

void CompressWeights::compute_svd(at::Tensor matrix)
{
    float* data=matrix.data_ptr<float>();
    Eigen::Map<Eigen::MatrixXf> e_matrix(data, matrix.size(0), matrix.size(1));
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(e_matrix, ComputeThinU | ComputeThinV);
    Eigen::MatrixXf U = svd.matrixU();

    int k=floor(this->percentage*U.cols());
    U = U(Eigen::all, Eigen::seq(0, k));
    Eigen::MatrixXf sigma = svd.singularValues().asDiagonal();
    sigma=sigma.topLeftCorner(k+1, k+1);
    Eigen::MatrixXf v = svd.matrixV().transpose();
    v=v(Eigen::seq(0,k), Eigen::seq(0,k));
    Eigen::MatrixXf N =sigma*v; 
    this->U=U;
    this->N=N;
}

std::vector<at::Tensor> CompressWeights::edit_weights(std::vector<at::Tensor> &weights){
    
}
