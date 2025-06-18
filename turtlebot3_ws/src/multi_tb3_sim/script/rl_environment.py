#!/usr/bin/env python3
"""
Multi-Agent Reinforcement Learning Environment for 3 TurtleBot3 robots
Integrates with OpenAI Gym and CityFlow for traffic-aware navigation
"""

import gym
from gym import spaces
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, Imu
import threading
import time
from typing import Dict, List, Tuple, Any, Optional
import cityflow
import cv2
import json
import math
from collections import deque
import yaml

class TurtleBot3MultiAgentEnv(gym.Env):
    """
    Multi-agent gym environment for 3 TurtleBot3 robots with CityFlow integration
    """
    
    def __init__(self, config_file: Optional[str] = None):
        super(TurtleBot3MultiAgentEnv, self).__init__()
        
        # Initialize ROS2
        if not rclpy.ok():
            rclpy.init()
        self.node = Node('turtlebot3_rl_env')
        
        # Robot configuration
        self.num_robots = 3
        self.robot_names = ['tb3_0', 'tb3_1', 'tb3_2']
        
        # Initialize CityFlow if config provided
        self.use_cityflow = config_file is not None
        if self.use_cityflow:
            try:
                self.cityflow_eng = cityflow.Engine(config_file, thread_num=1)
                self.node.get_logger().info("CityFlow initialized successfully")
            except Exception as e:
                self.node.get_logger().warn(f"CityFlow initialization failed: {e}")
                self.use_cityflow = False
        
        # Environment parameters
        self.max_episode_steps = 1000
        self.current_step = 0
        self.episode_rewards = {name: 0.0 for name in self.robot_names}
        
        # QoS profile for reliable communication
        self.qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        # Define spaces
        self.setup_spaces()
        
        # ROS2 interfaces
        self.setup_ros_interfaces()
        
        # State variables
        self.robot_states = {}
        self.laser_data = {}
        self.imu_data = {}
        self.goals = {}
        
        # Initialize robot states
        self.reset_robot_states()
        
        # Spinning thread for ROS2
        self.spin_thread = threading.Thread(target=self.spin_ros, daemon=True)
        self.spin_thread.start()
        
        # Wait for initial data
        self.node.get_logger().info("Waiting for initial robot data...")
        self.wait_for_initial_data()
        
    def setup_spaces(self):
        """Define action and observation spaces"""
        # Action space: [linear_velocity, angular_velocity] for each robot
        # TurtleBot3 limits: linear ~0.22 m/s, angular ~2.84 rad/s
        self.action_space = spaces.Dict({
            robot_name: spaces.Box(
                low=np.array([-0.22, -2.84]),
                high=np.array([0.22, 2.84]),
                dtype=np.float32
            ) for robot_name in self.robot_names
        })
        
        # Observation space components:
        # - Robot pose: x, y, theta (3)
        # - Robot velocity: linear, angular (2)
        # - Laser scan: 360 points downsampled to 36 (36)
        # - Goal relative position: dx, dy, distance, angle (4)
        # - Other robots relative positions: 2 * (dx, dy, distance) (6)
        # - Traffic density (if CityFlow enabled): (1)
        obs_dim = 3 + 2 + 36 + 4 + 6 + (1 if self.use_cityflow else 0)
        
        self.observation_space = spaces.Dict({
            robot_name: spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_dim,),
                dtype=np.float32
            ) for robot_name in self.robot_names
        })
        
    def setup_ros_interfaces(self):
        """Setup ROS2 publishers and subscribers"""
        self.cmd_pubs = {}
        self.odom_subs = {}
        self.laser_subs = {}
        self.imu_subs = {}
        
        for robot_name in self.robot_names:
            # Command velocity publishers
            self.cmd_pubs[robot_name] = self.node.create_publisher(
                Twist, f'/{robot_name}/cmd_vel', self.qos_profile
            )
            
            # Odometry subscribers
            self.odom_subs[robot_name] = self.node.create_subscription(
                Odometry, f'/{robot_name}/odom',
                lambda msg, name=robot_name: self.odom_callback(msg, name),
                self.qos_profile
            )
            
            # Laser scan subscribers
            self.laser_subs[robot_name] = self.node.create_subscription(
                LaserScan, f'/{robot_name}/scan',
                lambda msg, name=robot_name: self.laser_callback(msg, name),
                self.qos_profile
            )
            
            # IMU subscribers
            self.imu_subs[robot_name] = self.node.create_subscription(
                Imu, f'/{robot_name}/imu',
                lambda msg, name=robot_name: self.imu_callback(msg, name),
                self.qos_profile
            )
            
        self.node.get_logger().info("ROS2 interfaces setup complete")
        
    def reset_robot_states(self):
        """Initialize robot states"""
        for robot_name in self.robot_names:
            self.robot_states[robot_name] = {
                'position': np.array([0.0, 0.0, 0.0]),  # x, y, theta
                'velocity': np.array([0.0, 0.0]),       # linear, angular
                'last_update': time.time()
            }
            self.laser_data[robot_name] = np.zeros(360)
            self.imu_data[robot_name] = np.array([0.0, 0.0, 0.0])  # ax, ay, az
            
            # Set random goals within reasonable bounds
            goal_x = np.random.uniform(-3.0, 3.0)
            goal_y = np.random.uniform(-3.0, 3.0)
            self.goals[robot_name] = np.array([goal_x, goal_y])
            
    def spin_ros(self):
        """Spin ROS2 in separate thread"""
        while rclpy.ok():
            try:
                rclpy.spin_once(self.node, timeout_sec=0.01)
            except Exception as e:
                self.node.get_logger().error(f"Error in ROS spin: {e}")
                break
                
    def wait_for_initial_data(self, timeout=10.0):
        """Wait for initial data from all robots"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            all_ready = True
            for robot_name in self.robot_names:
                if time.time() - self.robot_states[robot_name]['last_update'] > 5.0:
                    all_ready = False
                    break
            if all_ready:
                self.node.get_logger().info("All robots ready!")
                return
            time.sleep(0.1)
        self.node.get_logger().warn("Timeout waiting for robot data")
        
    def odom_callback(self, msg: Odometry, robot_name: str):
        """Callback for odometry data"""
        pos = msg.pose.pose.position
        orient = msg.pose.pose.orientation
        
        # Convert quaternion to euler angle (yaw)
        siny_cosp = 2 * (orient.w * orient.z + orient.x * orient.y)
        cosy_cosp = 1 - 2 * (orient.y * orient.y + orient.z * orient.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        # Update position
        self.robot_states[robot_name]['position'] = np.array([pos.x, pos.y, yaw])
        
        # Update velocity
        linear_vel = math.sqrt(msg.twist.twist.linear.x**2 + msg.twist.twist.linear.y**2)
        angular_vel = msg.twist.twist.angular.z
        self.robot_states[robot_name]['velocity'] = np.array([linear_vel, angular_vel])
        self.robot_states[robot_name]['last_update'] = time.time()
        
    def laser_callback(self, msg: LaserScan, robot_name: str):
        """Callback for laser scan data"""
        # Convert to numpy and handle inf values
        ranges = np.array(msg.ranges)
        ranges[np.isinf(ranges)] = msg.range_max
        ranges[np.isnan(ranges)] = msg.range_max
        self.laser_data[robot_name] = ranges
        
    def imu_callback(self, msg: Imu, robot_name: str):
        """Callback for IMU data"""
        self.imu_data[robot_name] = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ])
        
    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[Dict, Dict, Dict, Dict]:
        """Execute one step in the environment"""
        # Send actions to robots
        for robot_name, action in actions.items():
            if robot_name in self.robot_names:
                cmd_msg = Twist()
                cmd_msg.linear.x = float(np.clip(action[0], -0.22, 0.22))
                cmd_msg.angular.z = float(np.clip(action[1], -2.84, 2.84))
                self.cmd_pubs[robot_name].publish(cmd_msg)
        
        # Step CityFlow if enabled
        if self.use_cityflow:
            try:
                self.cityflow_eng.next_step()
            except Exception as e:
                self.node.get_logger().warn(f"CityFlow step failed: {e}")
        
        # Wait for sensor updates
        time.sleep(0.1)
        
        # Get observations
        observations = self.get_observations()
        
        # Calculate rewards
        rewards = self.calculate_rewards(actions)
        
        # Check if done
        dones = self.check_done()
        
        # Additional info
        infos = self.get_info()
        
        self.current_step += 1
        
        return observations, rewards, dones, infos
        
    def get_observations(self) -> Dict[str, np.ndarray]:
        """Get current observations for all robots"""
        observations = {}
        
        # Get traffic density if CityFlow is enabled
        traffic_density = 0.0
        if self.use_cityflow:
            try:
                vehicles = self.cityflow_eng.get_vehicles()
                traffic_density = len(vehicles) / 100.0  # Normalize
            except:
                traffic_density = 0.0
        
        for robot_name in self.robot_names:
            robot_state = self.robot_states[robot_name]
            position = robot_state['position']
            velocity = robot_state['velocity']
            goal = self.goals[robot_name]
            
            # Robot pose and velocity
            pose_vel = np.concatenate([position, velocity])
            
            # Downsample laser scan (360 -> 36 points)
            laser_full = self.laser_data[robot_name]
            laser_downsampled = laser_full[::10]  # Every 10th point
            laser_downsampled = np.clip(laser_downsampled, 0, 3.5)  # Clip to reasonable range
            
            # Goal relative position
            goal_dx = goal[0] - position[0]
            goal_dy = goal[1] - position[1]
            goal_distance = math.sqrt(goal_dx**2 + goal_dy**2)
            goal_angle = math.atan2(goal_dy, goal_dx) - position[2]
            # Normalize angle to [-pi, pi]
            goal_angle = math.atan2(math.sin(goal_angle), math.cos(goal_angle))
            goal_info = np.array([goal_dx, goal_dy, goal_distance, goal_angle])
            
            # Other robots relative positions
            other_robots_info = []
            for other_name in self.robot_names:
                if other_name != robot_name:
                    other_pos = self.robot_states[other_name]['position']
                    dx = other_pos[0] - position[0]
                    dy = other_pos[1] - position[1]
                    distance = math.sqrt(dx**2 + dy**2)
                    other_robots_info.extend([dx, dy, distance])
            other_robots_info = np.array(other_robots_info)
            
            # Combine all observations
            obs_components = [pose_vel, laser_downsampled, goal_info, other_robots_info]
            
            if self.use_cityflow:
                obs_components.append(np.array([traffic_density]))
                
            obs = np.concatenate(obs_components)
            observations[robot_name] = obs.astype(np.float32)
            
        return observations
        
    def calculate_rewards(self, actions: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate rewards for each robot"""
        rewards = {}
        
        for robot_name in self.robot_names:
            position = self.robot_states[robot_name]['position']
            goal = self.goals[robot_name]
            action = actions.get(robot_name, np.array([0.0, 0.0]))
            
            # Goal distance reward
            goal_distance = np.linalg.norm(position[:2] - goal)
            goal_reward = -goal_distance * 0.5
            
            # Goal reached bonus
            goal_bonus = 10.0 if goal_distance < 0.3 else 0.0
            
            # Collision avoidance (laser-based)
            min_laser_dist = np.min(self.laser_data[robot_name])
            collision_penalty = 0.0
            if min_laser_dist < 0.2:
                collision_penalty = -5.0
            elif min_laser_dist < 0.5:
                collision_penalty = -1.0 * (0.5 - min_laser_dist)
            
            # Action smoothness (encourage smooth movements)
            action_penalty = -0.01 * (abs(action[0]) + abs(action[1]))
            
            # Multi-robot coordination (bonus for maintaining distance)
            coordination_bonus = 0.0
            for other_name in self.robot_names:
                if other_name != robot_name:
                    other_pos = self.robot_states[other_name]['position']
                    robot_distance = np.linalg.norm(position[:2] - other_pos[:2])
                    if 0.8 < robot_distance < 2.0:  # Ideal cooperation distance
                        coordination_bonus += 0.1
                    elif robot_distance < 0.5:  # Too close penalty
                        coordination_bonus -= 0.5
            
            # Living penalty (encourage task completion)
            living_penalty = -0.01
            
            total_reward = (goal_reward + goal_bonus + collision_penalty + 
                          action_penalty + coordination_bonus + living_penalty)
            
            rewards[robot_name] = total_reward
            self.episode_rewards[robot_name] += total_reward
            
        return rewards
        
    def check_done(self) -> Dict[str, bool]:
        """Check if episode is done for each robot"""
        dones = {}
        episode_done = self.current_step >= self.max_episode_steps
        
        for robot_name in self.robot_names:
            position = self.robot_states[robot_name]['position']
            goal = self.goals[robot_name]
            
            # Goal reached
            goal_reached = np.linalg.norm(position[:2] - goal) < 0.3
            
            # Collision detected
            collision = np.min(self.laser_data[robot_name]) < 0.15
            
            # Individual done condition
            robot_done = goal_reached or collision or episode_done
            dones[robot_name] = robot_done
            
        # Add global done flag
        dones['__all__'] = episode_done or all(dones.values())
        
        return dones
        
    def get_info(self) -> Dict[str, Dict]:
        """Get additional information"""
        infos = {}
        
        for robot_name in self.robot_names:
            position = self.robot_states[robot_name]['position']
            goal = self.goals[robot_name]
            goal_distance = np.linalg.norm(position[:2] - goal)
            
            infos[robot_name] = {
                'goal_distance': goal_distance,
                'episode_reward': self.episode_rewards[robot_name],
                'position': position.tolist(),
                'goal': goal.tolist(),
                'step': self.current_step
            }
            
        return infos
        
    def reset(self) -> Dict[str, np.ndarray]:
        """Reset the environment"""
        # Stop all robots
        for robot_name in self.robot_names:
            cmd_msg = Twist()
            self.cmd_pubs[robot_name].publish(cmd_msg)
        
        # Reset episode variables
        self.current_step = 0
        self.episode_rewards = {name: 0.0 for name in self.robot_names}
        
        # Reset goals
        for robot_name in self.robot_names:
            goal_x = np.random.uniform(-3.0, 3.0)
            goal_y = np.random.uniform(-3.0, 3.0)
            self.goals[robot_name] = np.array([goal_x, goal_y])
        
        # Wait for stabilization
        time.sleep(1.0)
        
        # Return initial observations
        return self.get_observations()
        
    def close(self):
        """Clean up resources"""
        # Stop all robots
        for robot_name in self.robot_names:
            cmd_msg = Twist()
            self.cmd_pubs[robot_name].publish(cmd_msg)
        
        # Destroy ROS node
        if hasattr(self, 'node'):
            self.node.destroy_node()
        
        self.node.get_logger().info("Environment closed")

# Utility functions for creating configurations

def create_default_config():
    """Create default configuration for the environment"""
    config = {
        'max_episode_steps': 1000,
        'robot_names': ['tb3_0', 'tb3_1', 'tb3_2'],
        'goal_bounds': {
            'x_min': -3.0, 'x_max': 3.0,
            'y_min': -3.0, 'y_max': 3.0
        },
        'action_bounds': {
            'linear_max': 0.22,
            'angular_max': 2.84
        },
        'reward_weights': {
            'goal_distance': -0.5,
            'goal_bonus': 10.0,
            'collision_penalty': -5.0,
            'action_penalty': -0.01,
            'coordination_bonus': 0.1,
            'living_penalty': -0.01
        }
    }
    return config

def save_config(config, filename='turtlebot_rl_config.yaml'):
    """Save configuration to YAML file"""
    with open(filename, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

if __name__ == "__main__":
    # Example usage
    print("Initializing TurtleBot3 Multi-Agent RL Environment...")
    
    # Create environment
    env = TurtleBot3MultiAgentEnv()
    
    print("Environment created successfully!")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Test episode
    obs = env.reset()
    print(f"Initial observations shape: {[obs[name].shape for name in env.robot_names]}")
    
    # Random actions for testing
    for step in range(10):
        actions = {}
        for robot_name in env.robot_names:
            actions[robot_name] = env.action_space[robot_name].sample()
        
        obs, rewards, dones, infos = env.step(actions)
        print(f"Step {step}: Rewards = {rewards}")
        
        if dones['__all__']:
            break
    
    env.close()
    print("Test completed!")