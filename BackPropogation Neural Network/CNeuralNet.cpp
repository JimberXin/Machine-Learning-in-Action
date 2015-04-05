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

	if(inputs.size() != m_NumInputs)    return outputs;    //return empty vector

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

//*********************************8* Sigmoid function *****************************************
double CNeuralNet::Sigmoid(double netInput, double response){
	return  (1.0 / (1 +exp(-netInput/response)));
}



//************************************ Training Process *****************************************
bool CNeuralNet::Training(vector<vecDouble> &RealInput,  vector<vecDouble> &RealOutput, double LearningRate){
	   double error;
	   double errorSum = 0;
	   double weightAdd;

	   for(int i=0; i < RealInput.size(); ++i){
		   vector<WEIGHT_TYPE> outputs = CalcOutput(RealInput[i]);
		   if(outputs.empty())   return;

		   // update the output layer, for each neuron
		   for(int j=0; j < m_NumOutputs; ++j){
			    error = ((double)RealOutput[i][j] - outputs[j]) * outputs[j] * (1-outputs[j]);    //important!!!!
				errorSum += (RealOutput[i][j] - outputs[j]) *  (RealOutput[i][j] - outputs[j]);

				//for each weight of the outputLayer neuron j
				for(int k=0; k < m_vecLayers[m_NumHiddenLayers+1].m_vecNeurons[j].m_NumInputs-1; ++k){
					weightAdd = error * m_vecLayers[m_NumHiddenLayers].m_vecNeurons[j].output;
					m_vecLayers[m_NumHiddenLayers+1].m_vecNeurons[j].m_vecWeights[k] += weightAdd;
					                                                                                                                                                                                                            
				}


				// for each layer j , each neuron k, each weight of the hiddenLayer
				for(int j = m_NumHiddenLayers; j >=0; --j){

					//for each neuron k
					for(int k = 0; k < m_vecLayers[j].m_vecNeurons[k].m_NumInputs; ++k){

						// for each weight of the neuron k
						for(int m = 0; k < m_vecLayers[j].m_vecNeurons[k].
					}
				}

		   }
	   }
}
