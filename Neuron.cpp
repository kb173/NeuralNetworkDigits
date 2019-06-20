//
// Created by karl on 23.05.19.
//

#include "Neuron.h"

Neuron::Neuron(unsigned int outputCount) {
    for (int synapseNumber = 0; synapseNumber < outputCount; synapseNumber++) {
        outputs.emplace_back(Synapse());
    }
}
