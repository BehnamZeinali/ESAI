# ESAI


This repository presents the Python code for ESAI (Efficient Split Artificial Intelligence via Early Exiting Using Neural Architecture Search), as detailed in a paper submitted to IEEE Transaction on Emerging Topics in Computational Intelligence.

The repository consists of two main sections:

The first part involves the integration of the knowledge distillation technique into the search space of the Neural Architecture Search (NAS) morphism-based method proposed in [NAS-Morphism](https://arxiv.org/abs/1711.04528).

![KD_Nas](https://github.com/BehnamZeinali/ESAI/assets/29498989/1ddc7985-494b-4d9e-8b35-5a13ebc1df5e)



The second part includes the implementation of the proposed efficient split neural network framework. In this framework, a decision unit evaluates the inference results of the client model. If the uncertainty of the result is high, the sample is sent to a server for classification using a complex model with higher accuracy. Otherwise, the result is directly presented to the user.

![Framework](https://github.com/BehnamZeinali/ESAI/assets/29498989/5f787b9d-572a-409a-8fb0-8ed84b2548fd)

The repository contains the following files:

1- server_model.py: Implements a transfer learning method using a pre-trained model.

2- get_logits.py: Removes the last activation function layer and produces the logits of the server model.

3- distillation_nas.py: Integrates the knowledge distillation method with the NAS method presented in [NAS-Morphism](https://github.com/akwasigroch/NAS_network_morphism). The knowledge distillation technique implementation from [knowledge-distillation](https://github.com/TropComplique/knowledge-distillation-keras/tree/master) is utilized here.

4- decision_unit.py: Implements the proposed split AI framework.

![Figure 2024-04-30 135634](https://github.com/BehnamZeinali/ESAI/assets/29498989/cb8bc260-bcdc-4c47-b721-16649d7d16dd)



# References

[1] Geoffrey Hinton, Oriol Vinyals, Jeff Dean, [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)

[2] Thomas Elsken, Jan-Hendrik Metzen, Frank Hutter, [Simple And Efficient Architecture Search for Convolutional Neural Networks](https://arxiv.org/abs/1711.04528)

