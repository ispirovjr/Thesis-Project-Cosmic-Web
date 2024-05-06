# Thesis-Project-Cosmic-Web


This is a repo I will be using to store and share my bachelor thesis project code. 


## Problem Definition
Within cosmology the most used method for distance determination is redshift measurements, assuming the expansion of the universe is the primary cause of redshift. In reality, however that is not the case. Objects have velocity relative to the expanding universe and this is especially prevolent near massive objects, such as galaxy clusters, where the viral velocity can 'project' the objects in redshift space. This can lead to incredably wrong measurements that go against the Copernican princile such as the fingers of God. Resolving these errors is highly nontrivial and vital to understand the true structures in the universe. This can be shown via redshift distortions as shown in notebooks Initial and Secondary tasks. 


## Solution and Project
The goal of this project is to develop a machine learning algorythm that can reliably correct redshift distortions. To that end I will use the Illustris-3 Dark simulation for cosmological data. I distort the data using the innate velocity redshift as shown in early tasks and study the resultant structures of the cosmic web. To that end I will alsso develop density maps via [DTFE](https://github.com/jfeldbrugge/DTFE?tab=readme-ov-file). 

Moving further, I will use pytorch to teach a neural network how to reconstruct the true matter distribution from a flawed obervation.



DTFE code taken from https://github.com/jfeldbrugge/DTFE?tab=readme-ov-file
