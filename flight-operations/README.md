# Machine Learning-Enhanced Aircraft Landing Scheduling under Uncertainties
Contact: Yutian Pang, Arizona State University <br>
Email: yutian.pang@asu.edu

# Highlights
- We identify the problem of interest by investigating the various flight scenarios and observe that holding patterns exist in most of the arrival delay cases.
- We propose using machine learning to predict the estimated arrival time (ETA) distributions for landing aircraft from real-world flight recordings, which are further used to obtain the probabilistic Minimal Separation Time (MST) between successive arrival flights.
- We propose to incorporate the predicted MSTs into the constraints of the Time-Constrained Traveling Salesman Problem (TSP) for Aircraft Landing Scheduling (ALS) optimization. 
- We build a multi-stage conditional prediction algorithm based on the looping events, and find that explicitly including flight event counts and airspace complexity measures can benefit the model prediction capability. 
- We demonstrate that the proposed method reduces the total landing time with a controlled reliability level compared with the First-Come-First-Served (FCFS) rule by running experiments with real-world data.

## The code/data available in this work:
- [x] Data Engineering
- [x] Aircraft Landing Scheduling Case Studies
- [x] Optimization Solver for Time-Constrained Travelling Salesman Problem

## Citation
The manuscript has been accepted for publication by Transportation Research Part C. 
```
@article{pang2023machine,
  title={Machine Learning-Enhanced Aircraft Landing Scheduling under Uncertainties},
  author={Pang, Yutian and Zhao, Peng and Hu, Jueming and Liu, Yongming},
  journal={arXiv preprint arXiv:2311.16030},
  year={2023}
}
```
