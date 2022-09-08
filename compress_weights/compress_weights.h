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

    public:
        CompressWeights(double user_percent);
        at::Tensor eigenVectorToTorchTensor(Eigen::MatrixXf e);
        at::Tensor compute_U(at::Tensor matrix);
        at::Tensor compute_SIGMA(at::Tensor matrix);
        at::Tensor compute_V(at::Tensor matrix);
        std::vector<at::Tensor> edit_weights(std::vector<at::Tensor> &weights, int index);
};