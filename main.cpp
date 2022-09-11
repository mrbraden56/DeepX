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

void print_weights(std::vector<at::Tensor> weights){
    for (auto & matrix : weights) {
        std::cout<<"("<<matrix.size(1)<<", "<<matrix.size(0)<<")\n";
    }
}

std::vector<at::Tensor> get_weights(torch::jit::script::Module module){
    std::vector<at::Tensor> weights;
    for (const auto& pair : module.named_parameters()) {
        if(pair.name.back()!='s'){
            weights.push_back(pair.value);
        }
        else{
            //bias
        }
    }

    return weights;
}

int main(int argc, char *argv[])
{
    torch::jit::script::Module module;
    module=torch::jit::load("/mnt/c/Users/brade/Research/DeepX/mnist/model/mnist.pt");
    std::vector<at::Tensor> weights = get_weights(module);
    print_weights(weights);
    CompressWeights compress_weights(0.8);
    std::cout<<"Weights Compressing...\n";
    compress_weights.edit_weights(weights, 1);
    torch::NoGradGuard no_grad;
    torch::autograd::GradMode::set_enabled(false);
    std::cout<<weights[0].size(1)<<std::endl;
    for (const auto& p : module.named_parameters()) {
        // at::Tensor z = &p.value; // note that z is a Tensor, same as &p : layers->parameters
        if(p.name.back()!='s'){
            // TODO: have to add operator overload to change weights of module
            p.value=weights[0];
        }
        else{
            //z.uniform_(l, u);
        }
    }
    torch::autograd::GradMode::set_enabled(true);
    std::vector<at::Tensor> weights2 = get_weights(module);
    print_weights(weights2);

}