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
Folder Structure

All source scripts live in the main directory.
All unit tests live in tests/.

```
├── EDH_LEDH_kernel_PFF_comparison_on_SSM.py
├── EDH_LEDH_kernel_PFF_comparison_on_SSM.py
├── EDH_LEDH_kernel_PFF_comparison_on_SSM.py   # Part 1.3 experiment driver
├── kernel_PFF_Hu_21.py                        # Kernel PFF implementation (Hu 2021)
├── PFPF_Li_2017.py                            # EDH / LEDH flows (Li 2017)
├── PF.py                                      # Bootstrap particle filter
├── EKF_UKF_PF_comparison.py                   # EKF / UKF / PF comparison on SV-SSM
├── sv_ssm.py                                  # Nonlinear SV state-space model
├── EKF_UKF.py                                 # EKF and UKF implementations
├── lgssm_kalman_tf.py                         # Linear-Gaussian SSM + Kalman filter
└── tests/
    ├── test_lgssm_kalman.py
    ├── test_kernel_PFF.py
    ├── test_sv_ssm_filters.py
    └── ... (other unit tests)
```
