//
// Created by karl on 20.06.19.
//

#ifndef NEURALNETWORKDIGITS_NEURALNET_H
#define NEURALNETWORKDIGITS_NEURALNET_H

#include <vector>
#include "Neuron.h"


class NeuralNet {

public:
    /// Topology 5, 2, 1 results in a network with 5 input neurons, 2 hidden neurons, 1 output neuron
    NeuralNet(const std::vector<unsigned int>& topology);

    /// Pass data through the net
    void feedForward(const std::vector<double>& input);

    /// Get the current output neuron values
    std::vector<double> getResults();

    /// Adjust the weights based on the current output neuron values, compared to the expected values
    void backPropagate(std::vector<double> expected);

private:
    std::vector<std::vector<Neuron>> layers;

};


#endif //NEURALNETWORKDIGITS_NEURALNET_H
