#include "CNeuralNet.h"
using namespace std;

//******************************** Constructor of class CNeuralNet *******************************
CNeuralNet::CNeuralNet(){
	// create each layers of the net
	if(m_NumHiddenLayers > 0){
		// create the first hidden layer
		m_vecLayers.push_back(SNeuronLayer(m_NeuronsPerHiddenLayer, m_NumInputs));

		// create the rest of the hidden layer
		for(int i = 0; i < m_NumHiddenLayers-1; ++i){
			m_vecLayers.push_back(SNeuronLayer(m_NeuronsPerHiddenLayer, m_NeuronsPerHiddenLayer));
		}

		// create the output layer
		m_vecLayers.push_back(SNeuronLayer(m_NumOutputs, m_NeuronsPerHiddenLayer));
	}  
	else {   // no hidden layers, just create outputlayer
		m_vecLayers.push_back(SNeuronLayer(m_NumOutputs, m_NumInputs));
	}
}

//*********************** Giving new weights, update the net with the new ones ***************************
void CNeuralNet::UpdateWeights(vector<double> &weights){
	int weight = 0 ;

	// for each layer
	for(int i=0; i < m_NumHiddenLayers+1; ++i){

		// for each neuron
		for(int j=0; j < m_vecLayers[i].m_NumNeurons; ++j){

			// for each weights
			for(int k=0; k < m_vecLayers[i].m_vecNeurons[j].m_NumInputs; ++k){
				m_vecLayers[i].m_vecNeurons[j].m_vecWeights[k] = weights[weight++];
			}
		}
	}

}


//*********************** Giving the inputs, calculate the output of the output layer ************************
vector<WEIGHT_TYPE> CNeuralNet::CalcOutput(vector<WEIGHT_TYPE> &inputs){
	// stores the resultant outputs from each layer
	vector<OUTPUT_TYPE> outputs;
	int weight = 0;

	if(inputs.size() != m_NumInputs)
		return outputs;    //return empty vector

	// for each layer, update
	for(int i=0; i < m_NumHiddenLayers+1; ++i){
		 // from the 2nd(i=1) layer, the previous layer's input is the output of next layer
		if(i > 0)   inputs = outputs;   

		outputs.clear();
		weight = 0 ;

	 //for each neurons, calculate its output from the previous input
		for(int j=0; j < m_vecLayers[i].m_NumNeurons; ++j){
			 double netInput = 0;
			 int NumInputs = m_vecLayers[i].m_vecNeurons[j].m_NumInputs;
			 // for each weights
			 for(int k=0; k < NumInputs-1; ++k)
				 netInput += m_vecLayers[i].m_vecNeurons[j].m_vecWeights[k] * inputs[weight++];

			 netInput += m_vecLayers[i].m_vecNeurons[j].m_vecWeights[NumInputs-1] * m_bias;

			 outputs.push_back(Sigmoid(netInput, 2));
			 weight = 0;
		}
	}

	return outputs;
}

//************************* Sigmoid function *******************************
double CNeuralNet::Sigmoid(double netInput, double response){
	return  (1.0 / (1 +exp(-netInput/response)));
}
