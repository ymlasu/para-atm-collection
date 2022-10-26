# Machine Learning-Enhanced Aircraft Landing Scheduling: Prediction and Optimization
Contact: Yutian Pang, Arizona State University <br>
Email: yutian.pang@asu.edu

# Highlights
- We identify the problem of interest by investigating the various flight scenarios and observe that holding patterns exist in most of the arrival delay cases.
- We propose using machine learning to predict the estimated arrival time (ETA) distributions for landing aircraft from real-world flight recordings, which are further used to obtain the probabilistic Minimal Separation Time (MST) between successive arrival flights.
- We propose a Traveling Salesman Problem (TSP) setup for Aircraft Landing Scheduling (ALS) optimization, where the data-driven predictions are incorporated into safety-related constraints.
- We build a multi-stage prediction algorithm conditioned on looping events, and find that explicitly including flight event counts and airspace complexity measures can benefit the model prediction capability. 
- We demonstrate that the proposed method reduces the total landing time with a controlled reliability level compared with the First-Come-First-Served (FCFS) rule by running experiments with real-world data.

# The paper is under review. The details will be released once the paper is pending online.

## The code/data used in this work:
- [x] Data Engineering
- [x] Aircraft Landing Scheduling Case Studies
- [x] Optimization Solver for Time Constrained Travelling Salesman Problem
 
## Citation
