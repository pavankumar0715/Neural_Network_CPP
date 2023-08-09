#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <iostream>
#include <vector>

using namespace std;

class Neuron
{
public:
	Neuron(int n_weights);
	~Neuron(){};

	void activate(vector<float> inputs);
	void transfer();
	float transfer_derivative() { return static_cast<float>(m_output * (1.0 - m_output)); };
	vector<float> &get_weights() { return m_weights; };
	float get_output() { return m_output; };
	float get_activation() { return m_activation; };
	float get_delta() { return m_delta; };
	void set_delta(float delta) { m_delta = delta; };

private:
	size_t m_nWeights;
	vector<float> m_weights;
	float m_activation;
	float m_output;
	float m_delta;
	void initWeights(int n_weights);
};

class Layer
{
public:
	Layer(int n_neurons, int n_weights);
	~Layer(){};

	vector<Neuron> &get_neurons() { return m_neurons; };

private:
	void initNeurons(int n_neurons, int n_weights);

	vector<Neuron> m_neurons;
};

class Network
{
public:
	Network();
	~Network(){};

	void initialize_network(int n_inputs, int n_hidden, int n_outputs);
	void add_layer(int n_neurons, int n_weights);
	vector<float> forward_propagate(vector<float> inputs);
	void backward_propagate_error(vector<float> expected);
	void update_weights(vector<float> inputs, float l_rate);
	void train(vector<vector<float>> trainings_data, float l_rate, size_t n_epoch, size_t n_outputs);
	int predict(vector<float> input);

private:
	size_t m_nLayers;
	vector<Layer> m_layers;
};

#endif
