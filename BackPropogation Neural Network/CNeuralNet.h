#ifndef _CNEURALNET_H
#define _CNEURALNET_H
#include<math.h>
#include<vector>

using namespace std;

typedef double INTPUT_TYPE;
typedef double WEIGHT_TYPE;
typedef double OUTPUT_TYPE;

inline double RandFloat(){ return (rand())/(RAND_MAX+1.0); }

inline double RandomClamped(){ return (RandFloat() - RandFloat()) ; }

// ******************************a single neuron cell***************************
struct SNeuron{
	 int m_NumInputs;   //how much input for the neuron cell
	 vector<WEIGHT_TYPE>  m_vecWeights;   // each weight of  the input
	 SNeuron(int NumInputs);   //constructor
};

SNeuron::SNeuron(int NumInputs):m_NumInputs(NumInputs+1){
	// for each weights, initialized them with random
	for(int i=0; i<NumInputs+1;++i){
		m_vecWeights.push_back(RandomClamped());
	}
}


//*********************************** a single neuron layer *****************************
struct SNeuronLayer{
	int m_NumNeurons;  //how much neurons of this layer
	vector<SNeuron> m_vecNeurons;   // the current layer
	SNeuronLayer(int NumNeurons, int  NumInputsPerNeuron);
};

SNeuronLayer::SNeuronLayer(int NumNeurons, int NumInputsPerNeuron):m_NumNeurons(NumNeurons){
	 
}
class CNeuralNet{
private:
	int  m_NumInputs;     //number of inputs
	int  m_NumOutputs;
	int  m_NumHiddenLayers;
	int  m_NeuronsPerHiddenLayer;
	const int m_bias = 1;
	vector<SNeuronLayer> m_vecLayers;

public:
	CNeuralNet();
	void InitNeual();
	// get all the weights from the net
	vector<double> GetWeights() const;    
	// replaces the weights with the new ones
	void UpdateWeights(vector<double> &weights);
	// calculates the ouputs from the input
	vector<WEIGHT_TYPE>  CalcOutput(vector<WEIGHT_TYPE> &inputs);
	// sigmoid activaction function
	inline double Sigmoid(double activaction, double response);
};


#endif