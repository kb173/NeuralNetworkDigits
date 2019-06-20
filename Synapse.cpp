//
// Created by karl on 20.06.19.
//

#include "Synapse.h"
#include "Util/Random.h"

double Synapse::getWeight() const {
    return weight;
}

void Synapse::setWeight(double weight) {
    Synapse::weight = weight;
}

double Synapse::getWeightChange() const {
    return weightChange;
}

void Synapse::setWeightChange(double weightChange) {
    Synapse::weightChange = weightChange;
}

Synapse::Synapse() {
    // Initialize with a random weight
    weight = Random::fromTo(0, 1);
    weightChange = 0;
}
