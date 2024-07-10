# Latent Factor Based Missing Link Recommender

This project implements a recommender system designed to predict the probability of missing links in a network. Given a social network with some known friendships, the goal is to estimate the probability of potential links where data is missing.

## Approach

### Data Representation

- The social network is represented as a matrix where each entry denotes the presence or absence of a friendship between two individuals. Missing entries are assumed to be zeros.

### Matrix Factorization

- We use Singular Value Decomposition (SVD) to decompose the friendship matrix M into three matrices: U, S, and V<sup>T</sup>, where M ≈ U × S × V<sup>T</sup>.
- The matrix S contains singular values that capture the strength of latent factors. Larger values in S correspond to more significant latent factors.

### Dimensionality Reduction

- To enhance the quality of recommendations, we remove columns and rows corresponding to smaller singular values in S. This process reduces the dimensionality of the matrices U, S, and V^T.

### Error Minimization

- After dimensionality reduction, we recombine the factor matrices to reconstruct the approximated friendship matrix.
- We use gradient descent to minimize the reconstruction error between the approximated matrix and the original matrix, with the exception of the missing entries.

### Link Prediction

- In the reconstructed matrix, missing entries are filled with real values representing the estimated probabilities of potential friendships. Higher values indicate a higher likelihood of a friendship.

## Usage

### Prepare Data

- The network.csv file contains a sample real-world network that can be used to test the program.

### Run the Program

- Use the missing_links.py script to process the data and perform link prediction.
