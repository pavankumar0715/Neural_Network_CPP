#include "NeuralNetwork.hpp"

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <algorithm>

using namespace std;

// Neuron Constructor
Neuron::Neuron(int n_weights)
{
	initWeights(n_weights);
	m_nWeights = n_weights;
	m_activation = 0;
	m_output = 0;
	m_delta = 0;
}

// Initialize weights
void Neuron::initWeights(int n_weights)
{
	for (int w = 0; w < n_weights; w++)
	{
		m_weights.push_back(float(rand()) / float(RAND_MAX));
	}
}

// Calculate the activation of a neuron for a given input
void Neuron::activate(vector<float> inputs)
{
	// the last weight is taken as bias
	m_activation = m_weights[m_nWeights - 1];
	for (size_t i = 0; i < m_nWeights - 1; i++)
	{
		m_activation += m_weights[i] * inputs[i];
	}
}

// Transfer the activation of the neuron to an actual output
void Neuron::transfer()
{
	m_output = 1.0f / (1.0f + exp(-m_activation));
}

// Layer Constructor
Layer::Layer(int n_neurons, int n_weights)
{
	initNeurons(n_neurons, n_weights);
}

void Layer::initNeurons(int n_neurons, int n_weights)
{
	for (int n = 0; n < n_neurons; n++)
	{
		m_neurons.push_back(Neuron(n_weights));
	}
}

// Network Constructor
Network::Network()
{
	srand(static_cast<unsigned int>(time(0)));
	m_nLayers = 0;
}

// Initialize a network manually
void Network::initialize_network(int n_inputs, int n_hidden, int n_outputs)
{
	add_layer(n_hidden, n_inputs + 1);
	add_layer(n_outputs, n_hidden + 1);
}

// Add another layer to the network
void Network::add_layer(int n_neurons, int n_weights)
{
	m_layers.push_back(Layer(n_neurons, n_weights));
	m_nLayers++;
}

// One forward propagation of an input
vector<float> Network::forward_propagate(vector<float> inputs)
{
	vector<float> new_inputs;
	for (size_t i = 0; i < m_nLayers; i++)
	{
		new_inputs.clear();

		vector<Neuron> &layer_neurons = m_layers[i].get_neurons();
		for (size_t n = 0; n < layer_neurons.size(); n++)
		{
			layer_neurons[n].activate(inputs);
			layer_neurons[n].transfer();
			new_inputs.push_back(layer_neurons[n].get_output());
		}
		inputs = new_inputs;
	}
	return inputs;
}

// Propagate the deviation from an expected output backwards through the network
void Network::backward_propagate_error(vector<float> expected)
{
	// reverse traverse the layers
	for (size_t i = m_nLayers; i-- > 0;)
	{
		vector<Neuron> &layer_neurons = m_layers[i].get_neurons();

		for (size_t n = 0; n < layer_neurons.size(); n++)
		{
			float error = 0.0;
			if (i == m_nLayers - 1)
			{
				error = expected[n] - layer_neurons[n].get_output();
			}
			else
			{
				for (auto &neu : m_layers[i + 1].get_neurons())
				{
					error += (neu.get_weights()[n] * neu.get_delta());
				}
			}
			layer_neurons[n].set_delta(error * layer_neurons[n].transfer_derivative());
		}
	}
}

// Update weights of a network after an error back propagation
void Network::update_weights(vector<float> inputs, float l_rate)
{
	for (size_t i = 0; i < m_nLayers; i++)
	{
		vector<float> new_inputs = {};
		if (i != 0)
		{
			for (auto &neuron : m_layers[i - 1].get_neurons())
			{
				new_inputs.push_back(neuron.get_output());
			}
		}
		else
		{
			new_inputs = vector<float>(inputs.begin(), inputs.end() - 1);
		}
		vector<Neuron> &layer_neurons = m_layers[i].get_neurons();

		for (size_t n = 0; n < layer_neurons.size(); n++)
		{
			vector<float> &weights = layer_neurons[n].get_weights();
			for (size_t j = 0; j < new_inputs.size(); j++)
			{
				weights[j] += l_rate * layer_neurons[n].get_delta() * new_inputs[j];
			}
			weights.back() += l_rate * layer_neurons[n].get_delta();
		}
	}
}

// Train the network with trainings data
void Network::train(vector<vector<float>> trainings_data, float l_rate, size_t n_epoch, size_t n_outputs)
{
	for (size_t e = 0; e < n_epoch; e++)
	{
		float sum_error = 0;

		for (const auto &row : trainings_data)
		{
			vector<float> outputs = forward_propagate(row);
			vector<float> expected(n_outputs, 0.0);
			expected[int(row.back())] = 1.0;
			for (size_t x = 0; x < n_outputs; x++)
			{
				sum_error += float(pow((expected[x] - outputs[x]), 2));
			}
			backward_propagate_error(expected);
			update_weights(row, l_rate);
		}
		cout << "> epoch=" << e << ", l_rate=" << l_rate << ", error=" << sum_error << endl;
	}
}

// Make a prediction for an input (one forward propagation)
int Network::predict(vector<float> input)
{
	vector<float> outputs = forward_propagate(input);
	return max_element(outputs.begin(), outputs.end()) - outputs.begin();
}
