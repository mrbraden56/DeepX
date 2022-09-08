#include <iostream>
#include <torch/script.h>
#include <torch/torch.h>
#include <thread>
#include <vector>
#include <typeinfo>
#include "compress_weights/compress_weights.h"
#include <cstdlib>
#include <memory>
#include <cxxabi.h>




std::string demangle(const char* name) {

    int status = -4; // some arbitrary value to eliminate the compiler warning

    // enable c++11 by passing the flag -std=c++11 to g++
    std::unique_ptr<char, void(*)(void*)> res {
        abi::__cxa_demangle(name, NULL, NULL, &status),
        std::free
    };

    return (status==0) ? res.get() : name ;
    // std::cout<<demangle(typeid(pair.value).name())<<std::endl;
}

int main(int argc, char *argv[])
{
    torch::jit::script::Module module;
    module=torch::jit::load("/mnt/c/Users/brade/Research/DeepX/mnist/model/mnist.pt");
    std::vector<at::Tensor> weights;
    std::vector<at::Tensor> bias;
    for (const auto& pair : module.named_parameters()) {
        if(pair.name.back()!='s'){
            weights.push_back(pair.value);
            std::cout<<"("<<pair.value.size(0)<<", "<<pair.value.size(1)<<")"<<std::endl;
        }
        else{
            bias.push_back(pair.value);
        }
    }
    std::cout<<"Note: that the weights are transposed so (64,784)->(784,64)"<<std::endl;
    CompressWeights compress_weights(0.2);
    compress_weights.edit_weights(weights, 1);
}