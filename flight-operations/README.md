# Aircraft Landing Scheduling with a Machine Learning-Enhanced Optimization Model: Investigations and Case Studies
Author: Yutian Pang, Arizona State University <br>
Email: yutian.pang@asu.edu

# Highlights
- We identify the problem of interests by investigating the various flight scenarios, and observe that holding patterns exist in most of the arrival delay cases.
- We propose using machine learning to predict the estimated arrival time (ETA) distributions for landing aircraft from real-world flight recordings, which are further used to obtain the Minimal Separation Time (MST) between successive arrival flights.
- We propose a Traveling Salesman Problem (TSP) setup for Aircraft Landing Scheduling (ALS), where the data-driven predictions are incorporated into safety-related constraints.
- We build a multi-stage prediction algorithm conditioned on looping event, and find that explicitly including flight event counts and airspace complexity measures can benefit the model prediction capability. 
- We demonstrate that the proposed method reduce the total landing time for all the landing aircraft in a given time window compared with the First-Come-First-Served (FCFS) rule, by running experiments with real-world data.


## The code for the multi-aircraft case study is shown in three parts:
- [x] Data Processing Demo
- [x] Aircraft Landing Scheduling Case Studies
- [x] TSP-TW Solver
 
## Citation
If you found this research useful to your work, please cite
