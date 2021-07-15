# UAS Obstacle Avoidance Integrating Safety Bound with Reinforcement Learning

Author: Jueming Hu, Arizona State University

Email: Jueming.Hu@asu.edu

This module demonstrates UAS obstacle avoidance with a risk-based safety bound using reinforcement learning.

The detailed information can be found [here](https://arc.aiaa.org/doi/abs/10.2514/6.2020-1372)
@inproceedings{hu2020uas,
  title={UAS Conflict Resolution Integrating a Risk-Based Operational Safety Bound as Airspace Reservation with Reinforcement Learning},
  author={Hu, Jueming and Liu, Yongming},
  booktitle={AIAA Scitech 2020 Forum},
  pages={1372},
  year={2020}
}

## Files

- main_RL_train_result.ipynb: main file for training and visualization

- boundSize.py: check potential collision among different shapes.

- multiPoly_bound.py: define RL environment, including transition model and reward function.

- plotting.py: visualization of RL training process
