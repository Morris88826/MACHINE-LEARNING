import torch
import torch.nn as nn

def max_pooling(input, kernel_size, stride):
    filter = nn.MaxPool2d(kernel_size, stride=stride)
    output = filter(input.view(1, 1, input.size(0), input.size(1)))
    output = output.view(output.size(2), output.size(3))

    return output

def convolution(input, kernel_size, padding=0, stride=1):
    # Get the size of the input and kernel
    filter = nn.Conv2d(1, 1, kernel_size, stride=stride, padding=padding)
    filter.weight.data = torch.Tensor([[1, 0, -1], [1, 0, -1], [1, 0, -1]]).view(1, 1, 3, 3)
    filter.bias.data = torch.Tensor([0])

    # Apply the filter to the input
    output = filter(input.view(1, 1, input.size(0), input.size(1)))
    output = output.view(output.size(2), output.size(3))

    return output

if __name__ == '__main__':
    input = torch.Tensor([[0, 2, 4, 1, 0],
                            [3, 1, 1, 0, 1],
                            [2, 4, 1, 0, 1],
                            [2, 0, 5, 2, 2],
                            [0, 1, 3, 2, 1]])

    
    
    print("Q1: Convolution with stride of 1")
    output = convolution(input, 3)
    print(output.detach().numpy())
    print("=====================================")

    print("Q2: Zero padding of 1 + convolution with stride of 1")
    output = convolution(input, 3, padding=1)
    print(output.detach().numpy())
    print("=====================================")

    print("Q3: Zero padding of 2 + convolution with stride of 2")
    output = convolution(input, 3, padding=2, stride=2)
    print(output.detach().numpy())
    print("=====================================")

    print("Q4: Convolution with stride of 1 + max pooling of 3 with stride of 1")
    output = convolution(input, 3)
    output = max_pooling(output, 3, 1)
    print(output.detach().numpy())
    print("=====================================")

    print("Q5: Zero padding of 2 + convolution with stride of 1 + max pooling of 3 with stride of 1")
    output = convolution(input, 3, padding=2)
    output = max_pooling(output, 3, 1)
    print(output.detach().numpy())
    print("=====================================")

