# Data-Driven Uncertainty Aware Multi-Aircraft Trajectory Prediction in the Near Terminal Airspace 
Author: Yutian Pang, Arizona State University <br>
Email: yutian.pang@asu.edu

## Bayesian Spatiotemporal Graph Transformer Network (B-STAR)
This work is following this [paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123570494.pdf) and extended into Bayesian sense for uncertainty quantification. This model is test to have state-of-the-art performance on the ETH/UCY dataset for pedestrain trajectory prediction, with reliable uncertainty estimates. Then we apply this framework to near-terminal multi-aircraft trajectory prediction task. 


## The code for the multi-aircraft case study is shown in three parts:
- [x] Part 1: IFF ASDE-X Data Processing Demo
- [x] Part 2: B-STAR Model Training Code
- [x] Part 3: Visualization Demo


## Near-Terminal Airspace Multi-Aircraft Trajectory Prediction 
- IFF ASDE-X data area of interests
  - 0.2 degrees latitude/longitude range near KATL [33.6366996, -84.4278640].
  - 8 hours duration each day since 2 pm.
  - Track points time interval 5s.
- Traning with data from Aug 1st, 2019 to Aug 6th, 2019. 
- Testing with data on Aug 7th, 2019.
- Neighboring aircrafts maximum radar separation limit 10km. 
- Observation/prediction time horizon ratio 3:2.


## Citation
If you found this research useful to your work, please cite this [paper](https://www.sciencedirect.com/science/article/pii/S0950705122004841):
```bib
@article{pang2022bayesian,
  title={Bayesian Spatio-Temporal grAph tRansformer network (B-STAR) for multi-aircraft trajectory prediction},
  author={Pang, Yutian and Zhao, Xinyu and Hu, Jueming and Yan, Hao and Liu, Yongming},
  journal={Knowledge-Based Systems},
  pages={108998},
  year={2022},
  publisher={Elsevier}
}
```
