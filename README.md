# Deep-Learning-for-Weather-and-Climate-Science

Nowcasting is weather forecasting on a very short term mesosacle period upto 6 hours. The goal is to give precise and timely prediction of precipitation, storm structure, hail potential, etc. in a local region over a short time period 
(eg., 0-6 hours). These predictions are useful for producing rainfall, storms, hails, etc alerts, providing weather guidance for airports, etc.

Weather Radar’s reflexivity is used by scientists to detect precipitation, evaluate storm structure, determine hail potential, etc. Sequence of radar reflexivity over a region for some time duration has spatiotemporal nature. Weather nowcasting is a spatiotemporal sequence forecasting problem with the sequence of past reflexivity maps as input and the
sequence of future reflexivity maps as output.

The LSTM encoder-decoder framework provides a general framework for sequence-to-sequence learning problems. I have implemented Convolutional LSTM Encoder-Decoder Network for weather forecasting with the sequences being maps of 
radar reflexivity.

