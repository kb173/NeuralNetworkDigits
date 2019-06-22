#include <iostream>
#include <mnist_reader.hpp>
#include <iomanip>
#include <map>
#include "NeuralNet.h"

// https://www.youtube.com/watch?v=KkwX7FkLfug 34:00

int main() {
    // Load MNIST data
    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
            mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(
                    "/home/karl/Data/Technikum/SEM4/MLE/NeuralNetworkDigits/Datasets");

    unsigned int inputNeuronCount = dataset.training_images.front().size();

    NeuralNet net = NeuralNet(std::vector<unsigned int>{inputNeuronCount, 3, 1});

    // Train
    for (unsigned int i = 0; i < dataset.training_images.size(); i++) {
        std::vector<double> trainingData;
        std::vector<double> expected = std::vector<double>{(double)dataset.training_labels[i] / 10.0};

        unsigned int trainingDataSize = dataset.training_images[i].size();

        for (unsigned int pixel = 0; pixel < trainingDataSize; pixel++) {
            double pixelValue = (double)dataset.training_images[i][pixel] / 255.0;
            trainingData.emplace_back(pixelValue);
        }

        net.feedForward(trainingData);
        net.backPropagate(expected);
    }

    // Classify!
    int goodDecisions = 0;
    int badDecisions = 0;

    std::map<std::string, std::map<std::string, int>> guessExpectMatrix;

    for (int i = 0; i < dataset.test_images.size(); i++) {
    }

    // Print the guessExpectMatrix
    for (int x = 0; x < 10; x++) {
        for (int y = 0; y < 10; y++) {
            std::cout << std::setw(10) << guessExpectMatrix[std::to_string(x)][std::to_string(y)];
        }

        std::cout << std::endl;
    }

    double accuracy = double(goodDecisions) / (goodDecisions + badDecisions);
    std::cout << "Accuracy: " << accuracy << std::endl;

    return 0;
}