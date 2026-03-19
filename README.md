## Team Members:

Annem Sai Reddy  
CB.SC.U4AIE24205  

Devana Madhavan Nambiar  
CB.SC.U4AIE24213  

Tharappel Manas  
CB.SC.U4AIE24257  

Zahwa K  
CB.SC.U4AIE24261  

## Semi-adaptive Synergetic Two-way Pseudoinverse Learning System

  Base paper : https://arxiv.org/pdf/2406.18931 
S2WPILS — Semi-adaptive Synergetic Two-way Pseudoinverse Learning System
A complete MATLAB implementation of the S2WPILS paper (Liu et al., 2024) with GPU acceleration, FISTA/ADMM solver comparison, and support for custom datasets.
## Requirements
MATLAB R2021a or later
Parallel Computing Toolbox (optional — for GPU acceleration)
No external toolboxes or .m helper files needed — everything is defined inside S2WPILS.m


## How to Run
Step 1 — Generate the custom dataset (optional)
matlabrun('generateDataset_01.m')
% Creates: myDataX.mat  (256 × 1000)
%          myDataY.mat  (1   × 1000)   classes: 1=zero, 2=one

## Step 2 — Open as a Live Script

MATLAB → File → New → Live Script → paste S2WPILS.m → Run

Step 3 — Set your dataset path (Section 1)
matlab
% Fashion-MNIST (default):
Label = importdata("...\Dataset\fashion_mnistnumY.mat");
X     = importdata("...\Dataset\fashion_mnistX.mat");

% Custom 0/1 dataset:
Label = importdata("...\Dataset\myDataY.mat");
X     = importdata("...\Dataset\myDataX.mat");

Step 4 — Set your split indices (Section 1)
matlab% Fashion-MNIST:
trainIdx = 1:60000;
valIdx   = 50001:60000;
testIdx  = 60001:70000;

% Custom dataset:
trainIdx = 1:700;
valIdx   = 701:850;
testIdx  = 851:1000;
Step 4 — Set your split indices (Section 1)
matlab% Fashion-MNIST:
trainIdx = 1:60000;
valIdx   = 50001:60000;
testIdx  = 60001:70000;

% Custom dataset:
trainIdx = 1:700;
valIdx   = 701:850;
testIdx  = 851:1000;


## Configuration 

All settings are defined in the `cfg` structure — no parameters are hidden elsewhere in the code.

| Parameter | Default | Description |
|----------|--------|-------------|
| `cfg.actFun` | `'tan'` | Activation function: `tan`, `sig`, `relu`, `prelu`, `gelu`, `sin`, `gau`, `mor` |
| `cfg.para` | `0.05` | Activation parameter |
| `cfg.lambda_cls` | `1e-12` | Classifier ridge regularization |
| `cfg.lambda_ft` | `1e-1` | Backward fine-tuning regularization |
| `cfg.MAX_SUBNET` | `5` | Number of parallel subsystems |
| `cfg.MAX_LAYER` | `3` | Maximum autoencoder layers per subsystem |
| `cfg.SAMPLE_RATIO` | `0.5` | Fraction of training data each subsystem sees |
| `cfg.AE_NEURONS` | `[2000,1500,500]` | Hidden neurons per autoencoder layer |
| `cfg.CLS_NEURONS` | `500` | Neurons in per-layer classifier |
| `cfg.FUSION_NEURONS` | `256` | Neurons in the final fused classifier |
| `cfg.earlyStopProb` | `0.99` | Probability threshold for accepting early stopping |
| `cfg.sparseMethod` | `'fista'` | Sparse solver used: `'fista'`, `'admm'`, or `'compare'` |
| `cfg.sparseIter` | `10` | Maximum iterations for sparse solvers |
| `cfg.sparseLambda` | `1e-3` | L1 regularization parameter for encoder initialization |
| `cfg.admm_rho` | `1.0` | ADMM penalty parameter ρ |
| `cfg.benchSubset` | `500` | Number of samples used for the FISTA vs ADMM benchmark |

## Recommended Settings for Custom 0/1 Dataset

For the synthetic binary digit dataset (0 vs 1), the following configuration provides stable training and good performance:

matlab

cfg.AE_NEURONS     = [128, 64, 32];
cfg.CLS_NEURONS    = 64;
cfg.FUSION_NEURONS = 256;
cfg.MAX_SUBNET     = 2;
cfg.SAMPLE_RATIO   = 0.8;

## Training Pipeline
Original Data
│
├── Random Sampling (per subsystem, ratio = SAMPLE_RATIO)
│
├── Forward Learning
│   ├── Stacked PILAE Layers
│   ├── Label-driven reverse pass (sparse AE init → FISTA / ADMM)
│   └── Pseudoinverse fine-tuning
│
├── Backward Learning
│
├── Feature Fusion
│   └── Concatenate forward + backward features
│
├── Hyperparameter Optimization
│   └── Grid search for best (l1, l2) layer combination
│
├── Classifier
│   └── Pseudoinverse-based (closed-form solution)
│
├── Iterative Subnet Training
│   └── Repeat for MAX_SUBNET times
│
├── Ensemble Learning
│   └── Sum aggregation (soft voting)
│
└── Final Prediction



## Local Functions

## Core Functions

| Function | Purpose |
|--------|--------|
| `trainSubsystem` | Trains one complete subsystem: forward learning → backward fine-tuning → feature fusion |
| `aeForward` | Forward pass through stacked autoencoders; saves `mapminmax` normalization parameters |
| `aeApply` | Applies the trained AE chain to validation/test data without refitting |
| `initInputWeight` | Initializes encoder weights using sparse coding (dispatches to **FISTA** or **ADMM**) |
| `trainSHLNN` | Trains a single hidden layer classifier using closed-form **pseudoinverse** solution |
| `testSHLNN` | Returns classification accuracy (fully GPU-native) |
| `testSHLNNFull` | Returns raw predictions along with classification accuracy |
| `backwardFinetune` | Performs backward learning using regularized pseudoinverse |
| `fusionFeatures` | Concatenates forward and backward path feature representations |
| `encodeBipolar` | Converts integer labels into `{−1, +1}` bipolar one-hot encoding |
| `activationFunc` | Implements all activation functions with full GPU compatibility |
| `deactivationFunc` | Approximate inverse activation functions used in the backward learning pass |
| `calcWeightsFISTA` | FISTA solver for **L1-regularized least squares** |
| `calcWeightsADMM` | ADMM solver for **L1-regularized least squares** |
| `fistaConvergenceCurve` | Generates per-iteration residual curve for FISTA convergence plotting |
| `admmConvergenceCurve` | Generates per-iteration residual curve for ADMM convergence plotting |
| `toGPU` | Moves arrays to GPU if `useGPU=true`; otherwise performs no operation |
| `canUseGPU` | Checks if Parallel Computing Toolbox and a compatible GPU are available |
| `ternary` | Inline string-based conditional helper used for benchmark output formatting |

## FISTA vs ADMM

Both **FISTA** and **ADMM** are used to solve the same **LASSO optimization problem** for initializing the encoder weights.

$$
\min_x \; \|WH - X\|^2 + \lambda \|x\|_r
$$

This formulation promotes **sparse encoder weights**, which improves feature extraction in the autoencoder layers.

### Solver Options

| Method | Description |
|------|-------------|
| `FISTA` | Fast Iterative Shrinkage-Thresholding Algorithm — fast convergence for many sparse problems |
| `ADMM` | Alternating Direction Method of Multipliers — more stable for certain ill-conditioned systems |
| `compare` | Runs both solvers and benchmarks convergence and training time |

### Running the Benchmark

Set the following configuration parameter:

matlab:
cfg.sparseMethod = 'compare';

## FISTA vs ADMM Comparison

| Aspect | FISTA | ADMM |
|------|------|------|
| **Strategy** | Gradient step + momentum + soft-threshold | Variable splitting with alternating **W / Z / U** updates |
| **Per-iteration cost** | 1 matrix multiply | 1 linear solve `(A'A + ρI) \ RHS` |
| **Convergence rate** | O(1/k²) | O(1/k) |
| **Key parameter** | Lipschitz constant `L = 1 / λ_max(A'A)` | Penalty parameter `ρ` (`cfg.admm_rho`, default = 1.0) |
| **GPU operations** | `*`, `eig`, `sign`, `max` | `*`, `\` (cuSolver), `sign`, `max` |

## GPU Support

The implementation automatically detects GPU availability.  
All heavy computations run on the **GPU**, while MATLAB functions that are **CPU-only** are temporarily gathered to the CPU, executed, and then moved back to the GPU.

| Operation | Device / Backend |
|----------|------------------|
| Matrix multiplies (`H = WI*X`, `WO = T*H'`) | GPU — **cuBLAS** |
| Linear solve in ADMM (`M \ RHS`) | GPU — **cuSolver** |
| Eigenvalue computation for Lipschitz constant (`eig`) | GPU — **cuSolver** |
| Element-wise activations (`tanh`, `exp`, `max`) | GPU — **CUDA element-wise kernels** |
| `orth()`, `mapminmax()`, `dividerand()` | CPU only |
| `zscore()` normalization | CPU only |

### GPU Handling Strategy

CPU-only functions follow this pattern:

matlab:
X_cpu = gather(X_gpu);   % move to CPU
X_cpu = mapminmax(X_cpu);
X_gpu = gpuArray(X_cpu); % move back to GPU

## Output

| Output | Description |
|------|-------------|
| **Console** | Displays per-subsystem validation accuracy and ensemble accuracy after each subsystem is added |
| **Plot 1** | Ensemble accuracy vs. number of subsystems with the best-performing point highlighted |
| **Plot 2** | FISTA vs ADMM convergence curves shown on a log-scale plot |
| **Plot 3** | Bar chart comparison of FISTA vs ADMM (training time, residual error, sparsity) |


<pre>
C5_MFC4_Semi-adaptive-Synergetic-Two-way-Pseudoinverse-Learning-System/
├── CODE/
│   ├── Dataset/
│   │   ├── mnistX.mat
│   │   └── mnistnumY.mat
│   ├── ActivationFunc.m
│   ├── DeactivationFunc.m
│   ├── calculateWeights4AE.m
│   ├── finetunning.m
│   ├── fusionnet.m
│   ├── initInputWeight.m
│   ├── PILAE.m
│   ├── S2WPILS_demo_MNIST.m
│   ├── targetPrepro.m
│   ├── train_SHLNN.m
│   └── test_SHLNN.m
├── README.md
└── .git/
└── PPT

</pre>


