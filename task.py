import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None,
        init_angle_velocities=None, runtime=10., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])
        self.last_pos = np.array([0., 0., 10.])

        # Debugging
        self.rewards = []
        self.position = []

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward()
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat)

        # Debugging
        self.rewards = []
        self.position = []

        return state

    def takeoff(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward_takeoff()
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)

        # Debugging
        self.rewards.append(reward)
        self.position.append(self.sim.pose)

        if self.sim.pose[2] >= self.target_pos[2]:
          done = True

        return next_state, reward, done

    def get_reward_takeoff(self):
        """Uses current pose of sim to return reward."""
        pose_diff = -abs(self.target_pos - self.sim.pose[:3])

        reward = 0

        # Reward if the copter gain altitude in last timestep. Penalize otherwise.
        if (self.target_pos[2] - self.sim.pose[2]) < (self.target_pos[2] - self.last_pos[2]):
          reward += 0.11
        else:
          reward -= 0.11

        # Additional reward if the copter crosses the target height.
        if self.sim.pose[2] >= self.target_pos[2]:
          reward += 1.0
        
        #if abs(self.target_pos[0] - self.sim.pose[0]) > abs(self.target_pos[0] - self.last_pos[0]):
        #  reward -= 0.1
        #if abs(self.target_pos[1] - self.sim.pose[1]) > abs(self.target_pos[1] - self.last_pos[1]):
        #  reward -= 0.1
        
        if np.sqrt((self.target_pos[0] - self.sim.pose[0])**2 + (self.target_pos[1] - self.sim.pose[1])**2) > 1:
            reward -= 1

        # Update the last position.
        self.last_pos = self.sim.pose[:3]

        return reward
