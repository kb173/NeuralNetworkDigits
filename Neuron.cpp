//
// Created by karl on 23.05.19.
//

#include <cmath>
#include "Neuron.h"

Neuron::Neuron(unsigned int outputCount, unsigned int indexInLayer) : indexInLayer(indexInLayer) {
    for (int synapseNumber = 0; synapseNumber < outputCount; synapseNumber++) {
        outputs.emplace_back(Synapse());
    }
}

double Neuron::getValue() const {
    return value;
}

void Neuron::setValue(double value) {
    Neuron::value = value;
}

void Neuron::activate(const std::vector<Neuron> &previousLayer) {
    double previousLayerOutput = 0.0;

    for (const Neuron& neuron : previousLayer) {
        previousLayerOutput += neuron.getValue() * neuron.outputs[indexInLayer].getWeight();
    }

    value = activationFunction(previousLayerOutput);
}

double Neuron::activationFunction(double value) {
    // Hyperbolic tangent
    return tanh(value);
}

double Neuron::activationFunctionDerivative(double value) {
    // Fast approximation of hyperbolic tangent function derivative
    return 1.0 - value * value;
}

void Neuron::calculateOutputGradients(double expected) {
    double delta = expected - value;

    gradient = delta * activationFunctionDerivative(value);
}

void Neuron::calculateHiddenGradients(const std::vector<Neuron> &nextLayer) {
    double dow = sumDOW(nextLayer);

    gradient = dow * activationFunctionDerivative(value);
}

void Neuron::updateWeights(std::vector<Neuron> &previousLayer) {
    for (Neuron &neuron : previousLayer) {
        Synapse &synapse = neuron.outputs[indexInLayer];

        double oldWeightChange = synapse.getWeightChange();
        double newWeightChange = learningRate * neuron.getValue() * gradient + momentum * oldWeightChange;

        synapse.setWeightChange(newWeightChange);
        synapse.setWeight(neuron.outputs[indexInLayer].getWeight() + newWeightChange);
    }
}

double Neuron::sumDOW(const std::vector<Neuron> &layer) const {
    double errorContributionSum = 0.0;

    unsigned int neuronCount = layer.size() - 1; // Subtract 1 for bias neuron

    for (unsigned int neuronNumber = 0; neuronNumber < neuronCount; neuronNumber++) {
        errorContributionSum += outputs[neuronNumber].getWeight() * layer[neuronNumber].gradient;
    }

    return errorContributionSum;
}
