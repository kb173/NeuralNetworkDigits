#include <iostream>
#include <mnist_reader.hpp>
#include <iomanip>
#include <map>
#include "NeuralNet.h"

std::vector<double> prepareImage(std::vector<uint8_t> data) {
    std::vector<double> doubleData;

    unsigned int trainingDataSize = data.size();

    for (unsigned int pixel = 0; pixel < trainingDataSize; pixel++) {
        double pixelValue = (double)data[pixel] / 128.0 - 1.0;
        doubleData.emplace_back(pixelValue);
    }

    return doubleData;
}

std::vector<double> prepareExpected(uint8_t value) {
    std::vector<double> expected;

    for (int j = 0; j < 10; j++) {
        if (j == value) {
            expected.push_back(1.0);
        } else {
            expected.push_back(0.0);
        }
    }

    return expected;
}

int main() {
    // Load MNIST data
    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
            mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(
                    "/home/karl/Data/Technikum/SEM4/MLE/NeuralNetworkDigits/Datasets");

    unsigned int inputNeuronCount = dataset.training_images.front().size();

    NeuralNet net = NeuralNet(std::vector<unsigned int>{inputNeuronCount, 48, 48, 10});

    // Train
    unsigned int trainingRuns = 300000;

    unsigned int trainingDataSize = dataset.training_images.size();

    for (unsigned int i = 0; i < trainingRuns; i++) {
        // Show progress
        if (i % 1000 == 999) {
            std::cout << (double)i * 100 / trainingRuns << "%" << std::endl;
        }

        unsigned int iFixed = i % trainingDataSize;

        net.feedForward(prepareImage(dataset.training_images[iFixed]));
        net.backPropagate(prepareExpected(dataset.training_labels[iFixed]));
    }

    std::cout << "Done training" << std::endl;

    // Classify!
    int goodDecisions = 0;
    int badDecisions = 0;

    std::map<unsigned int, std::map<unsigned int, unsigned int>> guessExpectMatrix;

    for (int i = 0; i < dataset.test_images.size(); i++) {
        net.feedForward(prepareImage(dataset.test_images[i]));

        // Get highest result value
        std::vector<double> results = net.getResults();
        double highest = 0;
        unsigned int result = 0;

        for (unsigned int j = 0; j < 10; j++) {
            if (results[j] > highest) {
                highest = results[j];
                result = j;
            }
        }

        if (result == dataset.test_labels[i]) {
            goodDecisions++;
        } else {
            badDecisions++;
        }

        guessExpectMatrix[result][dataset.test_labels[i]]++;
    }

    // Print the guessExpectMatrix
    for (unsigned int x = 0; x < 10; x++) {
        for (unsigned int y = 0; y < 10; y++) {
            std::cout << std::setw(10) << guessExpectMatrix[x][y];
        }

        std::cout << std::endl;
    }

    double accuracy = double(goodDecisions) / (goodDecisions + badDecisions);
    std::cout << "Accuracy: " << accuracy << std::endl;

    return 0;
}