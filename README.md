# Generative Artificial Intelligence Powered by Optimal Transport

This repository demonstrates the application of collisional optimal transport published in

- Sadr, Mohsen, and Hossein Gorji. "Collision-based Dynamics for Multi-Marginal Optimal Transport." arXiv preprint at [arXiv:2412.16385 (2024)](https://doi.org/10.48550/arXiv.2412.16385).

in calibrating diffusion models used in Generative AI. As an example, we compare our approach with the well-known Denoising Diffusion Probabilistic Models (DDPM)

- Ho, Jonathan, Ajay Jain, and Pieter Abbeel. "Denoising diffusion probabilistic models." Advances in neural information processing systems 33 (2020): 6840-6851, [arXiv.2006.11239](https://doi.org/10.48550/arXiv.2006.11239).

in generating images of MNIST, CIFAR10, and Food101.

For example, in case of Food101 dataset, after only 10 epochs, the Generative model trained on optimally paired samples of normal distribution and the dataset provides can generate more reasonable images compared to the DDPM counter part.

![Demo](testing/combined_Food101.png)

For trainig, please see examples in `training/` directory.

For testing the trained model, please see examples in `testing/` directory.
