#include <iostream>
#include <cmath>
#include "compress_weights.h"

CompressWeights::CompressWeights(double user_percent){
    this->percentage=user_percent;
}

at::Tensor CompressWeights::eigenVectorToTorchTensor(Eigen::MatrixXf e)
{
    auto t = torch::rand({e.rows(), e.cols()});
    float* data = t.data_ptr<float>();

    Eigen::Map<Eigen::MatrixXf> ef(data, t.size(0), t.size(1));
    ef = e.cast<float>();

    t.requires_grad_(true);
    return t;
}

at::Tensor CompressWeights::compute_U(at::Tensor matrix)
{
    float* data=matrix.data_ptr<float>();
    Eigen::Map<Eigen::MatrixXf> e_matrix(data, matrix.size(0), matrix.size(1));
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(e_matrix, ComputeThinU | ComputeThinV);
    Eigen::MatrixXf U = svd.matrixU();
    int k=floor(this->percentage*U.cols());
    U=U(Eigen::all, Eigen::seq(0, k));
    return eigenVectorToTorchTensor(U);
}

at::Tensor CompressWeights::compute_SIGMA(at::Tensor matrix)
{
    float* data=matrix.data_ptr<float>();
    Eigen::Map<Eigen::MatrixXf> e_matrix(data, matrix.size(0), matrix.size(1));
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(e_matrix, ComputeThinU | ComputeThinV);
    Eigen::MatrixXf U = svd.matrixU();
    int k=floor(this->percentage*U.cols());
    Eigen::MatrixXf sigma = svd.singularValues().asDiagonal();
    sigma=sigma.topLeftCorner(k+1, k+1);
    return eigenVectorToTorchTensor(sigma);
}

at::Tensor CompressWeights::compute_V(at::Tensor matrix)
{
    float* data=matrix.data_ptr<float>();
    Eigen::Map<Eigen::MatrixXf> e_matrix(data, matrix.size(0), matrix.size(1));
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(e_matrix, ComputeThinU | ComputeThinV);
    Eigen::MatrixXf U = svd.matrixU();
    int k=floor(this->percentage*U.cols());
    Eigen::MatrixXf v = svd.matrixV().transpose();
    v=v(Eigen::seq(0,k), Eigen::seq(0,k));
    return eigenVectorToTorchTensor(v);
}

void CompressWeights::edit_weights(std::vector<at::Tensor> &weights, int index){
    at::Tensor U = compute_U(weights[index]);
    at::Tensor SIGMA = compute_SIGMA(weights[index]);
    at::Tensor V = compute_V(weights[index]);
    at::Tensor N = SIGMA * V;
    at::Tensor U_next = compute_U(weights[index+1]);
    U=at::transpose(U, 0 ,1);
    weights[index]=U;
    weights[index+1]=U_next;
    weights.insert(weights.begin() + index+1, at::transpose(N, 0 ,1));
}
