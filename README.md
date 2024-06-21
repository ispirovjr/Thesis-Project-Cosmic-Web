# Thesis-Project-Cosmic-Web


This is a repository I will be using to store and share my bachelor thesis project code. 


## Problem Definition
Within cosmology the most used method for distance determination is measuring the redshift to galaxies, assuming the expansion of the universe is the primary cause of the redshift. In reality, however, that is not the case. Objects have velocity relative to the expanding universe and this is especially prevolent near massive objects, such as galaxy clusters, where the viral velocity can 'project' the objects in redshift space. This can lead to incredably wrong measurements that go against the Copernican princile such as the fingers of God. Resolving these errors is highly nontrivial and vital to understand the true structures in the universe. This can be shown via redshift distortions as shown in notebooks Initial and Secondary tasks. 


## Solution and Project
The goal of this project is to develop a machine learning algorythm that can reliably correct redshift distortions. To that end I will use the Illustris-3 Dark simulation for cosmological data. I distort the data using the innate velocity redshift as shown in early tasks and study the resultant structures of the cosmic web. To that end I will also develop density maps via [DTFE](https://github.com/jfeldbrugge/DTFE?tab=readme-ov-file). 

Using PyTorch I trained a network on the subhalo data. 

### Dataset Description

This dataset is designed to simulate galaxy formation within the ΛCDM (Lambda Cold Dark Matter) paradigm. Galaxies are formed within dark matter halos, driven by their gravitational potential. To simplify this complex process, we focus on dark matter subhalos instead of individual particles. This approach is both practical and efficient due to the following reasons:

- **Stability of Subhalos**: Unlike individual dark matter particles, which have low mass and high velocities, subhalos remain relatively stable even at high redshifts.
- **Reduced Computational Load**: Subhalos are fewer in number compared to individual particles, drastically reducing the required computational power.
- **Accurate Representation**: Individual particles can exhibit peculiar velocities, leading to problematic data such as negative distances. Subhalos provide a more reliable representation.

The dataset preparation involves:

1. **Random Point Selection**: A point is chosen at random within the simulation box to represent the observer's position.
2. **Coordinate Transformation**: The coordinates are transformed so the observer is at the origin, correcting the perspective.
3. **Application of Periodic Boundary Conditions**: This ensures the perspective remains consistent and avoids skewed data.

This methodology provides a streamlined and accurate dataset for studying galaxy formation in the context of dark matter subhalos.

### Neural Network Training and Dataset Description

The neural network model was designed to process and analyze subhalo data from Illustris 3 for studying galaxy formation within the ΛCDM paradigm.

#### Network Architecture

- **Input and Output Layers**: Fixed to reduce information load on the AI, focusing on corrected radial positions.
- **Fixed Input and Output Layers**: The architecture for these layers is predefined.
- **Soft Bottleneck Hidden Layers**: Inspired by Huppenkothen and Bachetti (2022), these layers condense and extrapolate information without being a true sequential bottleneck network.
- **Activation Functions**: Early Sigmoid functions were replaced to avoid unstable gradients, especially near the origin. ReLU activation is used to allow for a wider range of positive values.



#### Training Process

1. **Optimizers**:
   - **Stochastic Gradient Descent (SGD)**: Initially used for several thousand epochs until the model reaches a 20% accuracy benchmark.
   - **Adaptive Moment Estimation (Adam)**: Switched in after reaching the initial benchmark to maximize accuracy.
2. **Custom Loss Function**: 
   - **STandard dEviation Framework for Astrophysical Neural Networks (Stefann)**: Designed to force the neural network out of local minima during data fitting.

#### Dataset and Training Methodology

- **Training and Testing Datasets**: Various sizes tested, with 512 training samples found sufficient to avoid overfitting.
- **Overfitting Prevention**: Achieved without additional complex methods as long as the dataset exceeds 512 samples.
- **Model Validation**: Accuracy tested using a separate dataset to ensure generalization.

#### Bias and Data Receptivity

- **Grouped Subhalo Data**: Illustris 3 data is stored in a grouped manner, with subhalo members of superclusters appearing sequentially. This inherent pattern aids the neural network in learning, as neighboring neurons influence each other more effectively.
- **Pattern Relevance**: Discovered in collaboration with Aragon (2024), this bias is crucial for the model's performance.


#### Training Process

1. **Initial Training**:
   - **Stochastic Gradient Descent (SGD)**: Used for initial training until the model accuracy reaches about 20%. 
   - **Custom Loss Function (Stefann)**: Designed to avoid local minima and break out of the “Best Fit Sphere,” which is a local minimum where the model condenses predictions into a thin shell.
   
2. **Advanced Training**:
   - **Optimizer Switch to Adam**: After reaching 20% accuracy, SGD is swapped for Adam to leverage its capability in exploring the parameter space of a nearly saturated model. The learning rate is decreased to stabilize learning.
   - **Learning Rate Decay**: Introduced to further stabilize the training process.

3. **Training Stages**:
   - **Exploration Phase**: The network initially explores the parameter space with low accuracy and high loss.
   - **Best Fit Sphere**: The model tends to find a local minimum where it predicts a sphere of approximately half the universe's radius.
   - **Breaking Past the Sphere**: Using Stefann and SGD, the network can move past this stage and improve alignment with the data, though a central hole remains.
   - **Saturation and Refinement**: At around 20% accuracy, the optimizer is switched to Adam, which helps refine the model further.

#### Memory and Computational Considerations

- **GPU Utilization**: Training was optimized using GPUs from the Norma servers at the Kapteyn Institute, limited to 25 GB of memory.
- **Hidden Layer Size**: Adjusted to balance learning speed and memory constraints, with the realization that size affects speed but not the final accuracy.

#### Bias and Data Receptivity

- **Grouped Subhalo Data**: Subhalos are grouped by superclusters in memory, aiding the network in learning patterns due to the proximity of neurons influencing each other more effectively.
- **Bias Discovery**: Identified in collaboration with Aragon (2024), this grouping is vital for the network's performance.



## Visual Demonstrations

The training of this network was split into multiple steps and some architectural decisions were made due to issues realized while training the model. A scaled down version of the model was trained on Norma GPUs to demonstrate the importance of such features. The steps that will be shown here are described at length in Appendix A of my bachelor thesis.

The information will be presented in the form of animated gifs. A single dataset was used for a given example, meaaning the leftmost (redshift-distorted) and rightmost (physically distributed) perspectives don't change. The middle pannel shows the way the model corrects this testing dataset at a given epoch in training.

### Standard learning process

Firstly, the typical patterns of learning are shown. The AI begins by randomly flaining and exploring the parameter space. It eventually reaches the best fit sphere (will be shown again). This is a local minimum that the AI finds, explained more in the paper. Due to the use of the Stefann loss function, it breaks out, reaching a point of saturation, characterized by smooth edges and a cessation of learning.

![StefDirect](Executable/Model&#32;Figures/Gifs/LSES.gif)




