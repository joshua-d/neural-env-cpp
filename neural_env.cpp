//#include "stdafx.h"
#include "neural_env.h"
#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>

using namespace nenv;

int randint(int lower, int upper) {
	return rand() % (upper - lower + 1) + lower;
}

Neuron::Neuron(int layer, int id, double nbgvs = 0) {
	this->layer = layer;
	this->id = id;
	this->value = 0.5;
	this->weights = new vector<Weight*>;
	this->weighted = new vector<Neuron*>;
	this->bias = ((double)rand() / RAND_MAX) * nbgvs * 2 - nbgvs;
}

Neuron::~Neuron() {
	for (int i = 0; i < this->weights->size(); i++) {
		delete this->weights->at(i);
	}
	this->weights->clear();
	this->weighted->clear();

	delete this->weights;
	delete this->weighted;
}


Weight::Weight(Neuron* w_neuron, double value) {
	this->w_neuron = w_neuron;
	this->value = value;
}

Network::Network(NeuralEnv* nt) {
	this->nt = nt;
	this->neuron = new vector<vector<Neuron*>*>;

	this->subjectivity = (double)rand() / RAND_MAX + 1;
	this->ngs = randint(0, nt->output_neuron_amt);
	this->nrs = randint(0, nt->output_neuron_amt);
	this->wgs = randint(0, nt->output_neuron_amt);
	this->wrs = randint(0, nt->output_neuron_amt);
	this->nbgs = randint(0, nt->output_neuron_amt);
	this->nbrs = randint(0, nt->output_neuron_amt);
	this->wgvs = (double)rand() / RAND_MAX;
	this->nbgvs = (double)rand() / RAND_MAX;
	this->nbvs = (double)rand() / RAND_MAX;
	this->wvs = (double)rand() / RAND_MAX;

	this->fitness = 0;

	for (int i = 0; i < nt->layer_amt; i++) {
		this->neuron->push_back(new vector<Neuron*>);
	}

	for (int i = 0; i < nt->input_neuron_amt; i++) {
		this->neuron->at(0)->push_back(new Neuron(0, i));
	}

	for (int i = 0; i < nt->output_neuron_amt; i++) {
		this->neuron->at(nt->output_layer)->push_back(new Neuron(nt->output_layer, i));
	}
}

Network::~Network() {
	for (int i = 0; i < this->neuron->size(); i++) {
		for (int j = 0; j < this->neuron->at(i)->size(); j++) {
			delete this->get_neuron(i, j);
		}
		this->neuron->at(i)->clear();
		delete this->neuron->at(i);
	}

	this->neuron->clear();
	delete this->neuron;
}


void Network::input_data(vector<double> input) {
	for (int i = 0; i < this->nt->input_neuron_amt; i++) {
		this->neuron->at(0)->at(i)->value = input.at(i);
	}

	for (int i = 1; i < this->nt->layer_amt; i++) {
		for (int j = 0; j < this->neuron->at(i)->size(); j++) {
			double weight_total = 0;
			for (int k = 0; k < this->get_neuron(i, j)->weights->size(); k++) {
				weight_total += this->get_neuron(i, j)->weights->at(k)->w_neuron->value * this->get_neuron(i, j)->weights->at(k)->value;
			}
			weight_total += this->get_neuron(i, j)->bias;
			this->get_neuron(i, j)->value = 1 / (1 + pow(2.718, weight_total * (-1)));
		
			if (this->get_neuron(i, j)->value < this->nt->neuron_floor)
				this->get_neuron(i, j)->value = 0;
			else if (this->get_neuron(i, j)->value > this->nt->neuron_cap)
				this->get_neuron(i, j)->value = 1;
		}
	}
}

void Network::add_weight(Neuron* neuron, Neuron* w_neuron, double value) {
	for (int i = 0; i < neuron->weights->size(); i++) {
		if (neuron->weights->at(i)->w_neuron == w_neuron) {
			neuron->weights->at(i)->value = value;
			return;
		}
	}
	neuron->weights->push_back(new Weight(w_neuron, value));
	w_neuron->weighted->push_back(neuron);
}

void Network::generate_weight() {
	//TODO
	vector<Neuron*> neuron_list = this->get_neuron_list(1, this->nt->layer_amt);
	Neuron* neuron = neuron_list.at(randint(0, neuron_list.size() - 1));
	
	vector<Neuron*> w_neuron_list = this->get_neuron_list(0, neuron->layer);
	Neuron* w_neuron = w_neuron_list.at(randint(0, w_neuron_list.size() - 1));

	double value = (double)rand() / RAND_MAX * this->wgvs * 2 - this->wgvs;

	this->add_weight(neuron, w_neuron, value);
}

void Network::remove_weight() {
	vector<Neuron*> neuron_list = this->get_neuron_list(1, this->nt->layer_amt);
	Neuron* neuron = neuron_list.at(randint(0, neuron_list.size() - 1));

	while (neuron->weights->size() == 0) {
		if (this->has_no_weights())
			return;
		neuron = neuron_list.at(randint(0, neuron_list.size() - 1));
	}

	int weight_index = randint(0, neuron->weights->size() - 1);
	Weight* weight = neuron->weights->at(weight_index);

	for (int i = 0; i < weight->w_neuron->weighted->size(); i++) {
		if (weight->w_neuron->weighted->at(i) == neuron) {
			weight->w_neuron->weighted->erase(weight->w_neuron->weighted->begin() + i);
		}
	}

	delete neuron->weights->at(weight_index);

	neuron->weights->erase(neuron->weights->begin() + weight_index);

}

void Network::generate_neuron() {
	if (this->get_neuron_list(0, this->nt->layer_amt).size() == this->nt->input_neuron_amt + this->nt->max_h_neuron_amt * this->nt->max_hidden_layer_amt + this->nt->output_neuron_amt)
		return;
	int layer = randint(1, this->nt->max_hidden_layer_amt);
	while (this->neuron->at(layer)->size() == this->nt->max_h_neuron_amt) {
		layer = randint(1, this->nt->max_hidden_layer_amt);
	}
	this->neuron->at(layer)->push_back(new Neuron(layer, this->neuron->at(layer)->size()));
}

void Network::remove_neuron() {
	if (this->get_neuron_list(1, this->nt->output_layer).size() == 0)
		return;

	int layer = randint(1, this->nt->max_hidden_layer_amt);
	while (this->neuron->at(layer)->size() == 0)
		layer = randint(1, this->nt->max_hidden_layer_amt);
	int id = randint(0, this->neuron->at(layer)->size() - 1);

	Neuron* neuron = this->get_neuron(layer, id);

	//Get rid of weighted neurons' pointers to this one
	for (int i = 0; i < neuron->weights->size(); i++) {
		Weight* weight = neuron->weights->at(i);
		for (int j = 0; j < weight->w_neuron->weighted->size(); j++) {
			if (weight->w_neuron->weighted->at(j) == neuron) {
				weight->w_neuron->weighted->erase(weight->w_neuron->weighted->begin() + j);
				break;
			}
		}
	}

	//Get rid of weights on this neuron
	for (int i = 0; i < neuron->weighted->size(); i++) {
		Neuron* weighted = neuron->weighted->at(i);
		for (int j = 0; j < weighted->weights->size(); j++) {
			if (weighted->weights->at(j)->w_neuron == neuron) {
				delete weighted->weights->at(j);
				weighted->weights->erase(weighted->weights->begin() + j);
			}
		}
	}

	//Delete neuron from memory
	delete neuron;

	//Erase neuron from network and shift ids
	this->neuron->at(layer)->erase(this->neuron->at(layer)->begin() + id);
	for (int i = id; i < this->neuron->at(layer)->size(); i++) {
		this->get_neuron(layer, i)->id -= 1;
	}

}

void Network::generate_bias() {
	vector<Neuron*> neuron_list = this->get_neuron_list(1, this->nt->layer_amt);
	int neur = randint(0, neuron_list.size() - 1);
	neuron_list.at(neur)->bias = (double)rand() / RAND_MAX * this->nbgvs * 2 - this->nbgvs;
}

void Network::remove_bias() {
	vector<Neuron*> neuron_list = this->get_neuron_list(1, this->nt->layer_amt);
	int neur = randint(0, neuron_list.size() - 1);
	neuron_list.at(neur)->bias = 0;
}

vector<Neuron*> Network::get_neuron_list(int start_layer, int end_layer) {
	vector<Neuron*> neuron_list;
	for (int i = start_layer; i < end_layer; i++) {
		for (int j = 0; j < this->neuron->at(i)->size(); j++) {
			neuron_list.push_back(this->get_neuron(i, j));
		}
	}
	return neuron_list;
}

bool Network::has_no_weights() {
	for (int i = 1; i < this->nt->layer_amt; i++) {
		for (int j = 0; j < this->neuron->at(i)->size(); j++) {
			if (this->get_neuron(i, j)->weights->size() != 0)
				return false;
		}
	}
	return true;
}

Neuron* Network::get_neuron(int layer, int id) {
	return this->neuron->at(layer)->at(id);
}

double Network::get_neuron_value(int layer, int id) {
	return this->neuron->at(layer)->at(id)->value;
}

vector<Neuron*>* Network::get_output_neurons() {
	return this->neuron->at(this->nt->output_layer);
}



NeuralEnv::NeuralEnv(int pop_size, int input_neuron_amt, int max_h_neuron_amt, int output_neuron_amt, int max_hidden_layer_amt, void (*fitness_function)(Network*)) {
	this->pop_size = pop_size;
	this->input_neuron_amt = input_neuron_amt;
	this->max_h_neuron_amt = max_h_neuron_amt;
	this->output_neuron_amt = output_neuron_amt;
	this->max_hidden_layer_amt = max_hidden_layer_amt;

	this->fitness_function = fitness_function;

	this->cost_degree = 1;

	this->layer_amt = max_hidden_layer_amt + 2;
	this->output_layer = max_hidden_layer_amt + 1;

	this->inputs = new vector<vector<double>>;
	this->desired_outputs = new vector<vector<double>>;

	this->networks = new vector<Network*>;

	this->neuron_cap = 0.999999;
	this->neuron_floor = 0.000001;
	this->weight_cap = 10;
	this->weight_floor = -10;

	this->create_networks();
	this->initialize_networks();
}

void NeuralEnv::create_networks() {
	for (int i = 0; i < this->pop_size; i++)
		this->networks->push_back(new Network(this));
}

void NeuralEnv::initialize_networks() {
	for (int i = 0; i < this->pop_size; i++) {
		for (int j = 0; j < this->output_neuron_amt; j++) {
			if (randint(0, this->output_neuron_amt) < this->networks->at(i)->wgs)
				this->networks->at(i)->generate_weight();
			if (randint(0, this->output_neuron_amt) < this->networks->at(i)->nbgs)
				this->networks->at(i)->generate_bias();
		}
	}
}

Network* NeuralEnv::produce_child(Network* parent) {
	Network* child = new Network(this);

	//Subjectivities
	child->subjectivity = parent->subjectivity + (double)rand() / RAND_MAX * 2 - 1;
	child->ngs = parent->ngs + (double)rand() / RAND_MAX * parent->subjectivity * 2 - parent->subjectivity;
	child->nrs = parent->nrs + (double)rand() / RAND_MAX * parent->subjectivity * 2 - parent->subjectivity;
	child->wgs = parent->wgs + (double)rand() / RAND_MAX * parent->subjectivity * 2 - parent->subjectivity;
	child->wrs = parent->wrs + (double)rand() / RAND_MAX * parent->subjectivity * 2 - parent->subjectivity;
	child->nbgs = parent->nbgs + (double)rand() / RAND_MAX * parent->subjectivity * 2 - parent->subjectivity;
	child->nbrs = parent->nbrs + (double)rand() / RAND_MAX * parent->subjectivity * 2 - parent->subjectivity;
	child->wgvs = parent->wgvs + (double)rand() / RAND_MAX * parent->subjectivity * 2 - parent->subjectivity;
	child->nbgvs = parent->nbgvs + (double)rand() / RAND_MAX * parent->subjectivity * 2 - parent->subjectivity;
	child->nbvs = parent->nbvs + (double)rand() / RAND_MAX * parent->subjectivity * 2 - parent->subjectivity;
	child->wvs = parent->wvs + (double)rand() / RAND_MAX * parent->subjectivity * 2 - parent->subjectivity;

	//Subjectivity floors

	if (child->subjectivity < 1)
		child->subjectivity = 1;
	if (child->ngs < 0.1)
		child->ngs = 0.1;
	if (child->nrs < 0.1)
		child->nrs = 0.1;
	if (child->wgs < 0.1)
		child->wgs = 0.1;
	if (child->wrs < 0.1)
		child->wrs = 0.1;
	if (child->nbgs < 0.1)
		child->nbgs = 0.1;
	if (child->nbrs < 0.1)
		child->nbrs = 0.1;
	if (child->wgvs < 0.1)
		child->wgvs = 0.1;
	if (child->nbgvs < 0.1)
		child->nbgvs = 0.1;
	if (child->nbvs < 0.1)
		child->nbvs = 0.1;
	if (child->wvs < 0.1)
		child->wvs = 0.1;

	//Clear neurons
	for (int i = 1; i < this->layer_amt; i++) {
		for (int j = 0; j < child->neuron->at(i)->size(); j++) {
			delete child->get_neuron(i, j);
		}
		child->neuron->at(i)->clear();
	}

	//Weights and biases
	for (int i = 1; i < this->layer_amt; i++) {
		for (int j = 0; j < parent->neuron->at(i)->size(); j++) {

			child->neuron->at(i)->push_back(new Neuron(i, j));
			if (parent->get_neuron(i, j)->bias != 0)
				child->get_neuron(i, j)->bias = parent->get_neuron(i, j)->bias + (double)rand() / RAND_MAX * parent->nbvs * 2 - parent->nbvs;

			for (int k = 0; k < parent->get_neuron(i, j)->weights->size(); k++) {
				Weight* weight = parent->get_neuron(i, j)->weights->at(k);
				double weight_value = weight->value + (double)rand() / RAND_MAX * parent->wvs * 2 - parent->wvs;
				if (weight_value < this->weight_floor) {
					weight_value = this->weight_floor;
				}
				else if (weight_value > this->weight_cap) {
					weight_value = this->weight_cap;
				}

				child->add_weight(child->get_neuron(i, j), child->get_neuron(weight->w_neuron->layer, weight->w_neuron->id), weight_value);
			}

		}
	}

	//Mutations
	int weight_gen_amt = randint(0, (int)parent->wgs);
	int weight_rem_amt = randint(0, (int)parent->wrs);
	int neuron_gen_amt = randint(0, (int)parent->ngs);
	int neuron_rem_amt = randint(0, (int)parent->nrs);
	int bias_gen_amt = randint(0, (int)parent->nbgs);
	int bias_rem_amt = randint(0, (int)parent->nbrs);

	if (this->max_hidden_layer_amt != 0) {
		for (int i = 0; i < neuron_rem_amt; i++)
			child->remove_neuron();
		for (int i = 0; i < neuron_gen_amt; i++)
			child->generate_neuron();
	}

	for (int i = 0; i < weight_rem_amt; i++)
		child->remove_weight();
	for (int i = 0; i < weight_gen_amt; i++)
		child->generate_weight();

	for (int i = 0; i < bias_rem_amt; i++)
		child->remove_bias();
	for (int i = 0; i < bias_gen_amt; i++)
		child->generate_bias();

	return child;
}

void NeuralEnv::evaluate_fitnesses() {
	for (int i = 0; i < this->pop_size; i++) {
		this->fitness_function(this->networks->at(i));
	}
}

void nenv::evaluate_fitness(Network* network) {
	if (network->nt->inputs->size() != 0) {
		network->fitness = 0;
		for (int i = 0; i < network->nt->inputs->size(); i++) {
			network->input_data(network->nt->inputs->at(i));
			for (int j = 0; j < network->nt->output_neuron_amt; j++) {
				//std::cout << "abs " << abs(network->nt->desired_outputs->at(i).at(j) - network->get_neuron(network->nt->output_layer, j)->value) << std::endl;
				network->fitness += pow(abs(network->nt->desired_outputs->at(i).at(j) - network->get_neuron(network->nt->output_layer, j)->value), network->nt->cost_degree);
			}
		}
	}
}

bool nenv::compare_networks(Network* net_a, Network* net_b) {
	return net_a->fitness < net_b->fitness;
}

void NeuralEnv::reproduce() {
	this->evaluate_fitnesses();

	std::sort(this->networks->begin(), this->networks->end(), compare_networks);

	for (int i = 0; i < this->pop_size / 2; i++) {
		delete this->networks->at(i + this->pop_size / 2);
		this->networks->at(i + this->pop_size / 2) = this->produce_child(this->networks->at(i));
	}
}

void NeuralEnv::auto_reproduce(double fitness_threshold, int assure_amt, int log_interval) {
	bool ended = false;

	int log_counter = 1;
	int failure_cap = 10;
	int success_cap = 5;
	int failures = 0;
	int successes = 0;
	double failure_threshold = 0.5;
	double success_threshold = 0.5;
	double last_fitness = -1;

	std::cout << "Best fitness:" << std::endl;
	while (!ended) {
		this->reproduce();

		if (log_counter % log_interval == 0)
			std::cout << this->networks->at(0)->fitness << std::endl;
		log_counter++;

		if (this->networks->at(0)->fitness < 1 || this->networks->at(0)->fitness < fitness_threshold) {
			if (this->networks->at(0)->fitness < fitness_threshold && this->cost_degree == 1)
				ended = true;
			else if (this->networks->at(0)->fitness < success_threshold && this->cost_degree != 1) {
				std::cout << "Reproducing " << assure_amt << " times for assurance..." << std::endl;
				for (int i = 0; i < assure_amt; i++)
					this->reproduce();
				this->cost_degree--;
				std::cout << "Cost degree: " << this->cost_degree << std::endl;
				failures = 0;
			}
			else if (this->networks->at(0)->fitness < success_threshold) {
				success_threshold = fitness_threshold;
			}
		}

		if (last_fitness - this->networks->at(0)->fitness < failure_threshold) {
			failures++;
			successes = 0;
		}
		else {
			failures = 0;
			successes++;
		}

		if (failures >= failure_cap) {
			this->cost_degree++;
			std::cout << "Cost degree: " << this->cost_degree << std::endl;
			failures = 0;
		}
		else if (successes >= success_cap && this->cost_degree != 1) {
			this->cost_degree--;
			std::cout << "Cost degree: " << this->cost_degree << std::endl;
			successes = 0;
		}
	}
}

void NeuralEnv::reproduce_until(double max_fitness) {
	this->reproduce();
	while (this->networks->at(0)->fitness > max_fitness)
		this->reproduce();
}

void NeuralEnv::add_input_output(vector<double> input_list, vector<double> output_list) {
	this->inputs->push_back(input_list);
	this->desired_outputs->push_back(output_list);
}

Network* NeuralEnv::get_best_network() {
	this->evaluate_fitnesses();
	std::sort(this->networks->begin(), this->networks->end(), compare_networks);
	return this->networks->at(0);
}

