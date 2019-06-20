//
// Created by karl on 20.06.19.
//

#ifndef NEURALNETWORKDIGITS_SYNAPSE_H
#define NEURALNETWORKDIGITS_SYNAPSE_H


class Synapse {

public:
    Synapse();

    double getWeight() const;

    void setWeight(double weight);

    double getWeightChange() const;

    void setWeightChange(double weightChange);

private:
    double weight;
    double weightChange;
};


#endif //NEURALNETWORKDIGITS_SYNAPSE_H
