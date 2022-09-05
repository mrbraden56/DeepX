#include "../eigen-3.4.0/Eigen/Dense"
#include <torch/script.h>
#include <torch/torch.h>
using Eigen::MatrixXf;
using Eigen::ComputeThinU;
using Eigen::ComputeThinV;

class CompressWeights
{
    private:
        double percentage;
        Eigen::MatrixXf U;
        Eigen::MatrixXf N;

    public:
        CompressWeights(double user_percent);
        void compute_svd(at::Tensor matrix);
        std::vector<at::Tensor> edit_weights(std::vector<at::Tensor> &weights);
};