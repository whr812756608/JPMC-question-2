# Particle Flows for State-Space Models  
### Implementation for: **Part 1 – From classical filters to particle flows**

This repository implements the complete pipeline required in **Part 1** of the assignment:

1. **Classical filtering** for linear & nonlinear state-space models  
2. **EKF/UKF/PF** for nonlinear & non-Gaussian SSMs  
3. **Deterministic flows (EDH/LEDH)** and **kernel particle flows (Hu 2021)**  
4. **Comparison experiments:** RMSE, ESS, flow magnitude, Jacobian conditioning  
5. **Unit tests** for all major components  

All code is implemented from scratch — **no TFP particle-filter or LGSSM shortcuts**.

---

## 📦 Requirements

Create a file named `requirements.txt`:

```txt
matplotlib==3.10.7
numpy==2.3.5
pytest==8.4.2
scipy==1.16.3
tensorflow==2.19.1

