#include <iostream>
#include "NeuralNet.h"

// https://www.youtube.com/watch?v=KkwX7FkLfug 34:00

int main() {
    NeuralNet net = NeuralNet(std::vector<unsigned int>{2, 3, 1});

    std::cout << "Hello, World!" << std::endl;
    return 0;
}