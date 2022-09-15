from cProfile import run
import math
import torch
from torchvision import transforms, datasets
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import math

class Compress_Weights:

    def __init__(self, model: nn.Module, percent: float, layer: int, total_layers: int) -> None:
        self.model: nn.Module = model
        self.percent: float = percent
        self.network_layers=[]
        self.layer: int = layer
        self.k: int = 0
        self.total_layers: int = total_layers

    def compute_SVD(self, matrix: torch.Tensor)->torch.Tensor:
        U, S, V = torch.svd(matrix)
        self.k: int=math.floor(self.percent*U.size(1))
        U = U[:, :self.k]
        S=S[: self.k]
        V = V[: self.k, :]
        return U, S, V

    def run(self):
        self.model.requires_grad_(False)
        for i in range(1, self.total_layers+1):
            self.network_layers.append(eval(f"self.model.fc{i}"))
        U, S, V = self.compute_SVD(self.network_layers[self.layer].weight)
        U=torch.transpose(U, 0, 1)
        self.network_layers[self.layer].weight.resize_(self.k, 64)
        self.network_layers[self.layer].weight=nn.Parameter(U, requires_grad=True)
        self.network_layers[self.layer].bias.resize_(self.k)
        self.network_layers[self.layer].bias=nn.Parameter(S[:self.k], requires_grad=True)
        U, S, V = self.compute_SVD(self.network_layers[self.layer+1].weight)
        V=torch.transpose(V, 0, 1)
        self.network_layers[self.layer+1].weight.resize_(64, self.k)
        self.network_layers[self.layer+1].weight=nn.Parameter(V, requires_grad=True)
        self.model.requires_grad_(True)
        return self.model

class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(28*28, 64)
            self.fc2 = nn.Linear(64, 64)
            self.fc3 = nn.Linear(64, 64)
            self.fc4 = nn.Linear(64, 10)

        def forward(self, x):
            x = self.fc1(x)
            x = self.fc2(x)
            x = self.fc3(x)
            x = self.fc4(x)
            return F.log_softmax(x, dim=1)

def run_model(model):
    train = datasets.MNIST('', train=True, download=False,
                        transform=transforms.Compose([
                            transforms.ToTensor()
                        ]))

    trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1):
        for data in trainset:
            X, y = data
            model.zero_grad()
            output = model(X.view(-1,784))
            loss = loss_function(output, y)
            loss.backward()
            optimizer.step()
        print(loss)

    return model

def test_model(model):
    test = datasets.MNIST('', train=False, download=False,
                        transform=transforms.Compose([
                            transforms.ToTensor()
                        ]))

    testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=False)

    correct = 0
    total = 0

    with torch.no_grad():
        for data in testset:
            X, y = data
            output = model(X.view(-1,784))
            for idx, i in enumerate(output):
                if torch.argmax(i) == y[idx]:
                    correct += 1
                total += 1

    print("Accuracy: ", round(correct/total, 3))
    return model

def main():
    original_model = run_model(Net())
    test_model(original_model)
    compress_weights=Compress_Weights(model=original_model, 
                                      percent=0.2, 
                                      layer=1, 
                                      total_layers=4)
    compressed_model: nn.Module = compress_weights.run()
    test_model(compressed_model)
    compressed_model=run_model(compressed_model)
    test_model(compressed_model)

  
  
if __name__=="__main__":
    main()
