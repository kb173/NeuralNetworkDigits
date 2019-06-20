//
// Created by karl on 23.05.19.
//

#ifndef NEURALNETWORKDIGITS_NEURON_H
#define NEURALNETWORKDIGITS_NEURON_H

#include <vector>
#include "Synapse.h"

class Neuron {
public:
    explicit Neuron(unsigned int outputCount);

private:
    double value;
    std::vector<Synapse> outputs;
};


#endif //NEURALNETWORKDIGITS_NEURON_H
