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
If you found our work useful, please cite us.
```
@article{PANG2024104444,
  title = {Machine learning-enhanced aircraft landing scheduling under uncertainties},
  journal = {Transportation Research Part C: Emerging Technologies},
  volume = {158},
  pages = {104444},
  year = {2024},
  issn = {0968-090X},
  doi = {https://doi.org/10.1016/j.trc.2023.104444},
  url = {https://www.sciencedirect.com/science/article/pii/S0968090X23004345},
  author = {Yutian Pang and Peng Zhao and Jueming Hu and Yongming Liu},
  keywords = {Air traffic management, Landing scheduling, Data-driven prediction, Optimization, Machine learning},
}
```
