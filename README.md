# DGKL for Band Gap Prediction with M3Gnet Embeddings
The goal of this was to impelement DGKL on structure embeddings from M3GNet to predict band gaps of halide perovskites.

Main goals:
- learn a feature transformation on top of the precomputed embeddings
- compare DGKL vs Vanilla GPR
- Analyze the predictive uncertainty
- Extend model to multi-fidelity setting (PSE + HSE data)

## Dataset
Using strucured embeddings that are precomputed from the M3GNet
Each sample contanins:
- Composition
- Vector
- Fidelity
- Band_gap
Subset:
- PBE: 979 Samoles
- Multi-Fidelity : 1391 Samples (PBE+HSE)

## Models
### 1. DGKL
- Neutral feature extractor
- Variaitional Gaussian Process on latent space
- RBF kernel
- ELBO objective
### 2. GPR Baseline
- Standard Gaussian Process
- Uses raw embeddings

## Running
TBE

## Results
### DGKL (PBE-ONLY)
- Test RMSE: 0.22
- Test R^2: 0.98
### GPR 
- Test RMSE: 0.72
- Test R^2: 0.77
DGKL muchbetter than GPR which highlights that learned feature transformations are useful
### Multifidelity DGKL
- Test RMSE: 0.25
- TEST R^2: 0.97
Multifidelity model also does good but slightly worse the PBE-only, most likely due to task being more complex.

## Uncertainty
- need to impove calibration
- tune the kernel and likelihood parameters mode

## Repo Structure
