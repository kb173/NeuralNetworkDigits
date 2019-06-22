//
// Created by karl on 23.05.19.
//

#ifndef NEURALNETWORKDIGITS_NEURON_H
#define NEURALNETWORKDIGITS_NEURON_H

#include <vector>
#include "Synapse.h"

class Neuron {
public:
    explicit Neuron(unsigned int outputCount, unsigned int indexInLayer);

    void activate(const std::vector<Neuron> &previousLayer);

    double getValue() const;

    void setValue(double value);

    void calculateOutputGradients(double expected);

    void calculateHiddenGradients(const std::vector<Neuron> &previousLayer);

    void updateWeights(std::vector<Neuron> &previousLayer);

private:
    double value;
    double gradient;
    std::vector<Synapse> outputs;
    unsigned int indexInLayer;
    static double activationFunction(double value);
    static double activationFunctionDerivative(double value);
    double sumDOW(const std::vector<Neuron> &layer) const;
    double learningRate = 0.2;
    double momentum = 0.4;
};


#endif //NEURALNETWORKDIGITS_NEURON_H
