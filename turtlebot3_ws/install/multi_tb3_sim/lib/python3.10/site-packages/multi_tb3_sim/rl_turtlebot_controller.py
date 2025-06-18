#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
import numpy as np
import torch
from stable_baselines3 import PPO, A2C, SAC, TD3  # Import your specific algorithm
import math
from tf_transformations import euler_from_quaternion

class RLTurtleBotController(Node):
    def __init__(self, model_path, algorithm='PPO', robot_ids=['id01', 'id02', 'id03']):
        super().__init__('rl_turtlebot_controller')
        
        # Load the trained RL model
        self.algorithm = algorithm
        self.model = self.load_model(model_path, algorithm)
        self.robot_ids = robot_ids
        
        # Initialize robot states
        self.robot_states = {}
        self.publishers = {}
        self.subscribers = {}
        
        # Set up publishers and subscribers for each robot
        for robot_id in robot_ids:
            self.setup_robot_interface(robot_id, robot_id)
        
        # Control parameters
        self.control_frequency = 10.0  # Hz
        self.timer = self.create_timer(1.0/self.control_frequency, self.control_callback)
        
        # State space parameters (adjust based on your training environment)
        self.observation_size = 24  # Adjust based on your model's input
        self.action_size = 2  # [linear_velocity, angular_velocity]
        
        self.get_logger().info(f'RL TurtleBot Controller initialized with {algorithm} model')
        self.get_logger().info(f'Controlling robots: {robot_ids}')

    def load_model(self, model_path, algorithm):
        """Load the trained RL model"""
        try:
            if algorithm.upper() == 'PPO':
                model = PPO.load(model_path)
            elif algorithm.upper() == 'A2C':
                model = A2C.load(model_path)
            elif algorithm.upper() == 'SAC':
                model = SAC.load(model_path)
            elif algorithm.upper() == 'TD3':
                model = TD3.load(model_path)
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
            
            self.get_logger().info(f'Successfully loaded {algorithm} model from {model_path}')
            return model
        except Exception as e:
            self.get_logger().error(f'Failed to load model: {str(e)}')
            raise e

    def setup_robot_interface(self, robot_name, robot_id):
        """Set up publishers and subscribers for a single robot"""
        # Initialize robot state
        self.robot_states[robot_name] = {
            'position': {'x': 0.0, 'y': 0.0, 'theta': 0.0},
            'velocity': {'linear': 0.0, 'angular': 0.0},
            'laser_data': np.zeros(360),  # Assuming 360-degree laser
            'last_update': self.get_clock().now()
        }
        
        # Publishers
        cmd_topic = f'/{robot_name}/cmd_vel'
        self.publishers[robot_name] = self.create_publisher(Twist, cmd_topic, 10)
        
        # Subscribers
        odom_topic = f'/{robot_name}/odom'
        scan_topic = f'/{robot_name}/scan'
        
        self.subscribers[f'{robot_name}_odom'] = self.create_subscription(
            Odometry, odom_topic, 
            lambda msg, rname=robot_name: self.odom_callback(msg, rname), 10)
        
        self.subscribers[f'{robot_name}_scan'] = self.create_subscription(
            LaserScan, scan_topic,
            lambda msg, rname=robot_name: self.scan_callback(msg, rname), 10)

    def odom_callback(self, msg, robot_id):
        """Process odometry data"""
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        
        # Convert quaternion to euler angles
        _, _, yaw = euler_from_quaternion([
            orientation.x, orientation.y, orientation.z, orientation.w
        ])
        
        # Update robot state
        self.robot_states[robot_id]['position'] = {
            'x': position.x,
            'y': position.y,
            'theta': yaw
        }
        
        # Update velocity
        linear_vel = msg.twist.twist.linear
        angular_vel = msg.twist.twist.angular
        self.robot_states[robot_id]['velocity'] = {
            'linear': math.sqrt(linear_vel.x**2 + linear_vel.y**2),
            'angular': angular_vel.z
        }

    def scan_callback(self, msg, robot_id):
        """Process laser scan data"""
        # Convert laser scan to numpy array
        ranges = np.array(msg.ranges)
        
        # Handle infinite values
        ranges[np.isinf(ranges)] = msg.range_max
        ranges[np.isnan(ranges)] = 0.0
        
        # Downsample if necessary (adjust based on your training setup)
        if len(ranges) > 360:
            # Downsample to 360 points
            indices = np.linspace(0, len(ranges)-1, 360, dtype=int)
            ranges = ranges[indices]
        elif len(ranges) < 360:
            # Upsample to 360 points
            ranges = np.interp(np.linspace(0, len(ranges)-1, 360), 
                             np.arange(len(ranges)), ranges)
        
        self.robot_states[robot_id]['laser_data'] = ranges
        self.robot_states[robot_id]['last_update'] = self.get_clock().now()

    def get_observation(self, robot_id):
        """Convert robot state to observation vector for RL model"""
        state = self.robot_states[robot_id]
        
        # Basic observation: position, velocity, and processed laser data
        obs = []
        
        # Add position and velocity
        obs.extend([
            state['position']['x'],
            state['position']['y'],
            state['position']['theta'],
            state['velocity']['linear'],
            state['velocity']['angular']
        ])
        
        # Process laser data (take every 15th point for 24 laser readings)
        laser_data = state['laser_data']
        if len(laser_data) >= 360:
            # Sample 19 points from laser data (360/19 ≈ every 19th point)
            laser_indices = np.linspace(0, len(laser_data)-1, 19, dtype=int)
            laser_sample = laser_data[laser_indices]
            # Normalize laser readings (assuming max range of 10m)
            laser_sample = np.clip(laser_sample / 10.0, 0.0, 1.0)
            obs.extend(laser_sample.tolist())
        else:
            # If no laser data, pad with zeros
            obs.extend([0.0] * 19)
        
        # Ensure observation vector has correct size
        obs = obs[:self.observation_size]
        while len(obs) < self.observation_size:
            obs.append(0.0)
        
        return np.array(obs, dtype=np.float32)

    def action_to_twist(self, action):
        """Convert RL model action to ROS Twist message"""
        twist = Twist()
        
        # Assuming action is [linear_velocity, angular_velocity]
        # Adjust scaling based on your training environment
        if isinstance(action, np.ndarray):
            if len(action) >= 2:
                twist.linear.x = float(np.clip(action[0], -1.0, 1.0)) * 0.5  # Max 0.5 m/s
                twist.angular.z = float(np.clip(action[1], -1.0, 1.0)) * 1.0  # Max 1.0 rad/s
            else:
                # Single action case - might be discrete action space
                if action[0] == 0:  # Move forward
                    twist.linear.x = 0.3
                elif action[0] == 1:  # Turn left
                    twist.angular.z = 0.5
                elif action[0] == 2:  # Turn right
                    twist.angular.z = -0.5
                # action[0] == 3 would be stop (default zeros)
        
        return twist

    def control_callback(self):
        """Main control loop - called at regular intervals"""
        for robot_id in self.robot_ids:
            try:
                # Check if we have recent data for this robot
                if robot_id not in self.robot_states:
                    continue
                    
                # Get current observation
                observation = self.get_observation(robot_id)
                
                # Get action from RL model
                action, _ = self.model.predict(observation, deterministic=True)
                
                # Convert action to twist message
                twist_msg = self.action_to_twist(action)
                
                # Publish command
                self.publishers[robot_id].publish(twist_msg)
                
                # Log occasionally
                if robot_id == self.robot_ids[0] and self.get_clock().now().nanoseconds % 1000000000 < 100000000:  # Every ~1 second
                    self.get_logger().info(
                        f'Robot {robot_id}: Action=[{action[0]:.3f}, {action[1]:.3f}], '
                        f'Cmd=[{twist_msg.linear.x:.3f}, {twist_msg.angular.z:.3f}]'
                    )
                    
            except Exception as e:
                self.get_logger().error(f'Error controlling robot {robot_id}: {str(e)}')
                # Publish stop command on error
                stop_twist = Twist()
                if robot_id in self.publishers:
                    self.publishers[robot_id].publish(stop_twist)

def main(args=None):
    rclpy.init(args=args)
    
    # Configuration - modify these parameters
    import os
    MODEL_PATH = os.path.expanduser('~/cityflow_rl')  # Path to your trained model
    ALGORITHM = 'PPO'  # Change to your algorithm: PPO, A2C, SAC, TD3
    ROBOT_IDS = ['id01', 'id02', 'id03']  # Your specific robot IDs
    
    try:
        controller = RLTurtleBotController(MODEL_PATH, ALGORITHM, ROBOT_IDS)
        rclpy.spin(controller)
    except KeyboardInterrupt:
        print("Shutting down...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Send stop commands to all robots
        if 'controller' in locals():
            for robot_id in ROBOT_IDS:
                stop_msg = Twist()
                if robot_id in controller.publishers:
                    controller.publishers[robot_id].publish(stop_msg)
        
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()