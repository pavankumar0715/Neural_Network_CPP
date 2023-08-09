#include <iostream>
#include <vector>
#include <set>
#include <string>
#include <fstream>
#include <sstream>
#include <iterator>
#include <map>
#include <numeric>
#include <cmath>
#include "NeuralNetwork.hpp"

using namespace std;

vector<vector<float>> load_csv_data(string filename);
vector<float> evaluate_network(vector<vector<float>> dataset, int n_folds, float l_rate, int n_epoch, int n_hidden);
float accuracy_metric(vector<int> expect, vector<int> predict);

vector<float> parseLine(const string &line)
{
	stringstream ss(line);
	string str;
	vector<float> row;
	while (getline(ss, str, ','))
	{
		float data = stof(str);
		row.push_back(data);
	}
	if (getline(ss, str, '\n'))
	{
		float data = stof(str);
		row.push_back(data);
	}

	return row;
}

// Loading csv file and normalize the values
vector<vector<float>> load_csv_data(string filename)
{
	ifstream csv_file(filename);
	if (!csv_file)
	{
		cout << "Error opening file" << endl;
	}

	vector<vector<float>> data;

	string line;

	vector<float> mins;
	vector<float> maxs;
	bool first = true;

	while (getline(csv_file, line))
	{

		vector<float> row = parseLine(line);

		// To track min and max value of each column for normalization
		if (first)
		{
			mins = row;
			maxs = row;
			first = false;
		}
		else
		{
			for (size_t t = 0; t < row.size(); t++)
			{
				if (row[t] > maxs[t])
				{
					maxs[t] = row[t];
				}
				else if (row[t] < mins[t])
				{
					mins[t] = row[t];
				}
			}
		}

		data.push_back(row);
	}

	for (auto &vec : data)
	{
		for (size_t i = 0; i < vec.size() - 1; i++) // last column is output
		{
			vec[i] = (vec[i] - mins[i]) / (maxs[i] - mins[i]);
		}
	}

	return data;
}

vector<float> evaluate_network(vector<vector<float>> dataset, int n_folds, float l_rate, int n_epoch, int n_hidden)
{
	srand(static_cast<unsigned int>(time(0)));

	// Split dataset into folds

	vector<vector<vector<float>>> dataset_splits;

	vector<float> scores;

	size_t fold_size = u_int(dataset.size() / n_folds); // Could be millions of data points
	for (int f = 0; f < n_folds; f++)
	{
		vector<vector<float>> fold;
		while (fold.size() < fold_size)
		{
			int n = rand() % dataset.size(); // get a random index to remove it
			swap(dataset[n], dataset.back());
			fold.push_back(dataset.back());
			dataset.pop_back();
		}

		dataset_splits.push_back(fold);
	}

	// Iterate over folds with one as test and the rest as training sets
	for (size_t i = 0; i < dataset_splits.size(); i++)
	{
		vector<vector<vector<float>>> train_sets = dataset_splits;
		swap(train_sets[i], train_sets.back());
		vector<vector<float>> test_set = train_sets.back();
		train_sets.pop_back();
		vector<vector<float>> train_set;
		for (auto &s : train_sets)
		{
			for (auto &row : s)
			{
				train_set.push_back(row);
			}
		}
		vector<int> expected;
		for (auto &row : test_set)
		{
			expected.push_back(int(row.back()));
		}

		vector<int> predicted;

		set<float> expected_results;
		for (auto &r : train_set)
		{
			expected_results.insert(r.back());
		}
		int n_outputs = expected_results.size();
		int n_inputs = train_set[0].size() - 1;

		Network *network = new Network();
		network->initialize_network(n_inputs, n_hidden, n_outputs);
		network->train(train_set, l_rate, n_epoch, n_outputs);
		for (auto &row : test_set)
		{
			predicted.push_back(network->predict(row));
		}

		scores.push_back(accuracy_metric(expected, predicted));
	}

	return scores;
}

float accuracy_metric(vector<int> expect, vector<int> predict)
{
	int correct = 0;

	for (size_t i = 0; i < predict.size(); i++)
	{
		if (predict[i] == expect[i])
		{
			correct++;
		}
	}
	return float(correct * 100.0f / predict.size());
}

int main()
{
	vector<vector<float>> csv_data;
	csv_data = load_csv_data("seeds_dataset.csv");

	// Normalize the last column (turning the outputs into values starting from 0 for the one-hot encoding in the end)
	map<int, int> lookup = {};
	int index = 0;
	for (auto &vec : csv_data)
	{
		if (lookup.find(vec[vec.size() - 1]) == lookup.end())
		{
			lookup[vec[vec.size() - 1]] = index;
			vec[vec.size() - 1] = float(index);
			index++;
		}
		else
		{
			vec[vec.size() - 1] = float(lookup[vec[vec.size() - 1]]);
		}
	}

	int n_folds = 5;	 // how many folds you want to create from the given dataset
	float l_rate = 0.3f; // how much of an impact shall an error have on a weight
	int n_epoch = 500;	 // how many times should weights be updated
	int n_hidden = 5;	 // how many neurons you want in the first layer

	vector<float> scores = evaluate_network(csv_data, n_folds, l_rate, n_epoch, n_hidden);

	float sum_scores = 0.0;
	for (float &score : scores)
	{
		sum_scores += score;
	}
	float mean = sum_scores / float(scores.size());

	cout << "Mean accuracy: " << mean << endl;

	return 0;
}
