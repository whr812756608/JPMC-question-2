# Particle Flows for State-Space Models  
### Implementation for: **Part 1 – From classical filters to particle flows**

This repository implements the complete pipeline required in **Part 1** of the qusstion 2:

---

## Requirements

Create a file named `requirements.txt`:

```txt
matplotlib==3.10.7
numpy==2.3.5
pytest==8.4.2
scipy==1.16.3
tensorflow==2.19.1
```
Install dependencies:
```
pip install -r requirements.txt
```
## Folder Structure

All source scripts live in the main directory.
All unit tests live in tests/.

```
├── EDH_LEDH_kernel_PFF_comparison_on_SSM.py   # Main experiment for Part 2C
├── EKF_UKF.py                                 # EKF + UKF for nonlinear SSMs
├── EKF_UKF_PF_comparison.py                   # EKF/UKF vs PF comparison
├── PF.py                                      # Standard particle filter
├── PFPF_Li_2017.py                            # Invertible PF-PF (Li 2017)
├── kernel_PFF_Hu_21.py                        # Kernel embedded flow (Hu 2021)
├── lgssm_kalman_tf.py                         # Kalman filter + LGSSM simulation
└── sv_ssm.py                                  # Nonlinear Stochastic Volatility SSM
└── tests/                                     # all unit test
    ├── test_lgssm_kalman.py
    ├── test_EKF_UKF.py
    ├── test_sv_ssm.py
    └── ... (other unit tests)
```
## Script to each question

**1) I — Linear-Gaussian SSM + Kalman Filter**
| Scripts                       | Description                                                                                       |
| ---------------------------- | ------------------------------------------------------------------------------------------------- |
| `lgssm_kalman_tf.py`         | Problem a, b: LGSSM simulation + Kalman filter (no TFP). Joseph form, PSD checks, condition number diagnostics. |
| `tests/test_lgssm_kalman.py` | test files                                     |

**1) II — Nonlinear/Non-Gaussian SSM with EKF/UKF and Particle Filter**
| Scripts                        | Description                                                   |
| ------------------------------ | ------------------------------------------------------------- |
| `sv_ssm.py`                    | Problem a: Defines nonlinear/non-Gaussian stochastic volatility SSM.     |
| `EKF_UKF.py`                   | Problem b: EKF and UKF implementations.                                  |
| `PF.py`                        | Problem c: Bootstrap PF (systematic resampling).                         |
| `EKF_UKF_PF_comparison.py`     | Problem d: Benchmarks EKF vs UKF vs PF (RMSE, degeneracy, runtime, CPU RAM). |
| `tests/test_sv_ssm.py` |   test files                      |
| `tests/test_EKF_UKF.py` |  test files                        |
| `tests/test_PF.py` | test files                        |
| `tests/test_EKF_UKF_PF_comparison.py` |  test files                    |


**2) Deterministic and Kernel Flows**

| Script                                     | Description                                                               |
| ------------------------------------------ | ------------------------------------------------------------------------- |
| `PFPF_Li_2017.py`                          | Problem a: Deterministic EDH / LEDH particle flows.                                  |
| `kernel_PFF_Hu_21.py`                      | Problem b: Kernel-embedded particle flow (scalar + matrix-valued kernels).           |
| `EDH_LEDH_kernel_PFF_comparison_on_SSM.py` | Problem c: Full comparison on SV-SSM: RMSE, ESS, flow magnitude, Jacobian magnitude. |
| `tests/test_PFPF_Li_2017.py`                 | test files                |
| `tests/test_kernel_PFF_Hu_21.py`                 | test files              |
| `tests/test_EDH_LEDH_kernel_PFF_comparison_on_SSM.py`                 | test files                |

