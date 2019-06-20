//
// Created by karl on 20.06.19.
//

#include <random>
#include "Random.h"

double Random::fromTo(double from, double to) {
    std::uniform_real_distribution<double> unif(from, to);

    std::random_device rand_dev;
    std::mt19937 rand_engine(rand_dev());

    return unif(rand_engine);
}
