#pragma once
#include <vector>
#include <string>

using namespace std;

namespace nenv {

	struct Neuron {
		int layer;
		int id;
		double value;
		vector<struct Weight*>* weights;
		vector<struct Neuron*>* weighted;
		double bias;
		Neuron(int layer, int id, double nbgvs);
		~Neuron();
	};

	struct Weight {
		Neuron* w_neuron;
		double value;
		Weight(Neuron* w_neuron, double value);
	};

	struct Network {
		struct NeuralEnv* nt;
		vector<vector<Neuron*>*>* neuron;

		double subjectivity;
		double ngs;
		double nrs;
		double wgs;
		double wrs;
		double nbgs;
		double nbrs;
		double wgvs;
		double nbgvs;
		double nbvs;
		double wvs;

		double fitness;

		Network(NeuralEnv* nt);
		~Network();

		void input_data(vector<double> input);
		void add_weight(Neuron* neuron, Neuron* w_neuron, double value);
		void generate_weight();
		void remove_weight();
		void generate_neuron();
		void remove_neuron();
		void generate_bias();
		void remove_bias();
		vector<Neuron*> get_neuron_list(int start_layer, int end_layer);
		bool has_no_weights();
		Neuron* get_neuron(int layer, int id);
		double get_neuron_value(int layer, int id);
		vector<Neuron*>* get_output_neurons();
	};

	void evaluate_fitness(Network* network);
	bool compare_networks(Network* net_a, Network* net_b);

	struct NeuralEnv {
		int pop_size;
		int input_neuron_amt;
		int max_h_neuron_amt;
		int output_neuron_amt;
		int max_hidden_layer_amt;

		int cost_degree;

		int layer_amt;
		int output_layer;

		vector<vector<double>>* inputs;
		vector<vector<double>>* desired_outputs;

		vector<Network*>* networks;

		double neuron_cap;
		double neuron_floor;
		double weight_cap;
		double weight_floor;

		NeuralEnv(int pop_size, int input_neuron_amt, int max_h_neuron_amt, int output_neuron_amt, int max_hidden_layer_amt, void (*fitness_function)(Network*) = evaluate_fitness);

		void (*fitness_function)(Network* nt);

		void create_networks();
		void initialize_networks();
		void add_input_output(vector<double> input_list, vector<double> output_list);
		void evaluate_fitnesses();
		Network* produce_child(Network* parent);
		void reproduce();
		void auto_reproduce(double fitness_threshold=1, int assure_amt=10, int log_interval=1);
		void reproduce_until(double max_fitness);
		Network* get_best_network();
	};

}
