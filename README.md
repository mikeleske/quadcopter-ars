# quadcopter-ars

This repository implements the Augmented Random Search algorithm to let a simulated quadcopter learn to fly.
Paper: https://arxiv.org/abs/1803.07055

### Notes:
- **quadcopter-ars.ipynb** implements the algorithm, runs training and visualizes rewards
  - ars implmentation is taken from SuperDataScience course AI 2018 and modified to fit environment
- **task.py** implements the "agent" and its functionalities (e.g. reward system, operations) - Taken from Udacity Machine Learning course and modified
- **physics_sim.py** implement physics of quadcopter - Taken from Udacity Machine Learning course
