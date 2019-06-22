//
// Created by karl on 20.06.19.
//

#include <iostream>
#include <cmath>
#include "NeuralNet.h"

NeuralNet::NeuralNet(const std::vector<unsigned int>& topology) {
    unsigned int layerCount = topology.size();

    // Add layers
    for (unsigned int layerNumber = 0; layerNumber < layerCount; layerNumber++) {
        std::cout << "Adding layer" << std::endl;

        std::vector<Neuron> newLayer;

        // Add 1 for the bias neuron
        unsigned int neuronCount = topology[layerNumber] + 1;

        // Neurons in the next layer, so we know how many connections these will need
        unsigned int nextLayerNeuronCount = layerNumber == topology.size() - 1 ? 0 : topology[layerNumber + 1] + 1;

        // Add neurons within this layer
        for (unsigned int neuronNumber = 0; neuronNumber < neuronCount; neuronNumber++) {
            std::cout << "Adding neuron" << std::endl;

            newLayer.emplace_back(Neuron(nextLayerNeuronCount, neuronNumber));
        }

        // Bias neuron should always output 1.0
        newLayer.back().setValue(1.0);

        layers.emplace_back(newLayer);
    }
}

void NeuralNet::feedForward(const std::vector<double>& input) {
    unsigned int inputSize = input.size();
    unsigned int layerCount = layers.size();

    if (inputSize != layers[0].size() - 1) {
        std::cerr << "Input neurons don't match with the size of the data!" << std::endl;
    }

    // Assign the input values to the neurons in the first layer
    for (unsigned int inputNumber = 0; inputNumber < inputSize; inputNumber++) {
        layers[0][inputNumber].setValue(input[inputNumber]);
    }

    // Propagate forward through all other layers
    for (unsigned int layerNumber = 1; layerNumber < layerCount; layerNumber++) {
        unsigned int neuronCount = layers[layerNumber].size() - 1; // Subtract 1 for bias neuron

        // each neuron
        for (unsigned int neuronNumber = 0; neuronNumber < neuronCount; neuronNumber++) {
            const std::vector<Neuron> &previousLayer = layers[layerNumber - 1];

            layers[layerNumber][neuronNumber].activate(previousLayer);
        }
    }
}

void NeuralNet::backPropagate(const std::vector<double> &expected) {
    // Calculate overall error which will be minimized
    std::vector<Neuron> &outputLayer = layers.back();

    // Root mean square error
    overallError = 0.0;

    unsigned int outputLayerSize = outputLayer.size() - 1; // Subtract 1 for bias neuron

    for (unsigned int outputNumber = 0; outputNumber < outputLayerSize; outputNumber++) {
        double delta = expected[outputNumber] - outputLayer[outputNumber].getValue();

        overallError += delta * delta;
    }

    overallError /= outputLayerSize;
    overallError = sqrt(overallError);

    // Calculate output layer gradients
    for (unsigned int outputNumber = 0; outputNumber < outputLayerSize; outputNumber++) {
        outputLayer[outputNumber].calculateOutputGradients(expected[outputNumber]);
    }

    // Calculate hidden layer gradients from first hidden layer after output to last hidden layer before input
    for (unsigned int layerNumber = layers.size() - 2; layerNumber > 0; layerNumber--) {
        std::vector<Neuron> &currentLayer = layers[layerNumber];
        std::vector<Neuron> &nextLayer = layers[layerNumber + 1];

        unsigned int neuronCount = currentLayer.size();

        for (unsigned int neuronNumber = 0; neuronNumber < neuronCount; neuronNumber++) {
            currentLayer[neuronNumber].calculateHiddenGradients(nextLayer);
        }
    }

    // Update synapse weights for all layers from output to first hidden
    for (unsigned int layerNumber = layers.size() - 1; layerNumber > 0; layerNumber--) {
        std::vector<Neuron> &currentLayer = layers[layerNumber];
        std::vector<Neuron> &previousLayer = layers[layerNumber - 1];

        unsigned int neuronCount = currentLayer.size();

        for (unsigned int neuronNumber = 0; neuronNumber < neuronCount; neuronNumber++) {
            currentLayer[neuronNumber].updateWeights(previousLayer);
        }
    }
}

std::vector<double> NeuralNet::getResults() {
    std::vector<double> resultValues;
    std::vector<Neuron> &outputLayer = layers.back();

    unsigned int outputNeuronCount = outputLayer.size() - 1;

    for (unsigned int outputNeuronNumber = 0; outputNeuronNumber < outputNeuronCount; outputNeuronNumber++) {
        resultValues.push_back(outputLayer[outputNeuronNumber].getValue());
    }

    return resultValues;
}
