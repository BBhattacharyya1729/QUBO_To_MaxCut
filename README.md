# QUBO_To_Ising
Code for QUBO -> Ising Paper

### Utility Files
* **oputils.py**: Code for algebra with binary variables. Handles generation of QUBO problem instances

* **WarmStartUtils.py**: Code for GW_k and BM_k warmstarts. Includes vertex on top rotations (*TODO*: PSC warmstarts)

* **QAOAUtils.py**: QAOA utility functions. Includes both qiskit and custom simulator

### Problem Types
* Portfolio Optimization
* Random QUBO Discrete
* Random QUBO Continous 
* Travelling Salesman (Only Depth 0)
* Max Independent Set (gnp) (Only Depth 0)
* Max Independent Set (nws) (Only Depth 0)

### Data Generation
All probems have 2 files for data generation

* **Depth0.ipynb**: Samples 1000 Random instances and compares all warmstarts (GW3,GW2,BM3,BM2) with all vertex-on-top rotations

* **FullRun.ipynb**: Samples 10 instances from those generated in the depth0 files and compares "good" warmstarts (GW3,GW2) with rotations on first, last, and no qubits over depth 0-5.

* **PSC:** *TODO*, Samples 10 instances from those generated in the depth0 files and compares PSC warmstarts over depth 0-5.

All data is stored via pickle on [google drive](https://drive.google.com/drive/folders/1TCz_ncc0QwwceBb3bvijLQMW0Dol0Nef?usp=sharing) 

### Figures
All probems have 7 figures (as of now)
* Comparison_Boxplot: Compares Warmstart's relative errors and optimal sampling probability at depth 0 (all -1 rotations).

* Cost_Scatter: Plots the obtained cost vs optimal cost for each warmstart at Depth 0 (last plot is all -1 rotations)

* Full_Cost_Comparison: Plots the relative error vs depth averaged for the 10 instances selected in the FullRun file.

* Full_Prob_Comparison: Plots the log of 1 - optimal sampling probability averaged for the 10 instances selected in the FullRun file.

* Max_Cost_Hist: Plots how often each potential rotation Optimization maximized the cost at depth 0.


* Max_Cost_Hist: Plots how often each potential rotation Optimization maximized the optimal sampling probability at depth 0.

* Prob_BoxPlot: Boxplots of optimal sampling probability at depth0 for each warmstart with rotations on first,last, and no qubits (last plot is all -1 rotations).
