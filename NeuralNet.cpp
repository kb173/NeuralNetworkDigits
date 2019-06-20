//
// Created by karl on 20.06.19.
//

#include <iostream>
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
        unsigned int nextLayerNeuronCount = layerNumber == topology.size() ? 0 : topology[layerNumber + 1];

        // Add neurons within this layer
        for (unsigned int neuronNumber = 0; neuronNumber < neuronCount; neuronNumber++) {
            std::cout << "Adding neuron" << std::endl;

            newLayer.emplace_back(Neuron(nextLayerNeuronCount));
        }

        layers.emplace_back(newLayer);
    }
}

void NeuralNet::feedForward(const std::vector<double>& input) {
    if (input.size() != layers[0].size()) {
        std::cerr << "Input neurons don't match with the size of the data!" << std::endl;
    }
}
