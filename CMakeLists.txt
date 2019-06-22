cmake_minimum_required(VERSION 3.14)
project(NeuralNetworkDigits)

set(CMAKE_CXX_STANDARD 17)

include_directories(mnist)

add_executable(NeuralNetworkDigits main.cpp Neuron.cpp Neuron.h NeuralNet.cpp NeuralNet.h Synapse.cpp Synapse.h Util/Random.cpp Util/Random.h)