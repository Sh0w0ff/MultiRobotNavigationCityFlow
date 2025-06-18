#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import gym # Changed from gymnasium
import numpy as np
import cityflow
import json
import os
import math # For simple quaternion to yaw conversion

# --- Custom Gym Environment Definition ---
class CityFlowGymEnv(gym.Env):
    """
    Custom OpenAI Gym Environment that wraps CityFlow and communicates with TurtleBots via ROS2.
    Compatible with gym 0.26.2.
    """
    metadata = {'render_modes': ['human'], 'render_fps': 30} # Optional metadata for Gym

    def __init__(self, ros_node, num_turtlebots=3):
        super().__init__()
        self.ros_node = ros_node
        self.num_turtlebots = num_turtlebots

        # --- CityFlow Setup ---
        # Get the directory where the current Python script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_path = os.path.join(script_dir, "cityflow_config.json")
        self.roadnet_path = os.path.join(script_dir, "cityflow_roadnet.json")
        self.flow_path = os.path.join(script_dir, "cityflow_flow.json")

        self._create_dummy_cityflow_files() # Creates minimal CityFlow files

        try:
            self.cityflow_engine = cityflow.Engine(self.config_path, thread_num=1)
            self.ros_node.get_logger().info("CityFlow engine initialized within Gym environment.")
        except Exception as e:
            self.ros_node.get_logger().error(f"Failed to initialize CityFlow engine: {e}")
            raise RuntimeError("CityFlow engine initialization failed.") # Raise to stop node if critical

        self.cityflow_step_counter = 0 # Track CityFlow simulation steps

        # --- ROS2 TurtleBot Communication Setup ---
        self.cmd_vel_publishers = {}
        self.odom_subscribers = {}
        self.current_odom_data = {} # Store latest odom data for each TurtleBot

        for i in range(self.num_turtlebots):
            robot_id = f"tb3_{i}"
            # Publishers for cmd_vel
            self.cmd_vel_publishers[robot_id] = self.ros_node.create_publisher(Twist, f'/{robot_id}/cmd_vel', 10)
            # Subscribers for odom
            self.odom_subscribers[robot_id] = self.ros_node.create_subscription(
                Odometry,
                f'/{robot_id}/odom',
                lambda msg, rid=robot_id: self._odom_callback(msg, rid), # Use lambda for per-robot callback
                10
            )
            self.current_odom_data[robot_id] = None # Initialize to None

        # --- Define Gym Spaces ---
        # OBSERVATION SPACE: This will primarily come from CityFlow.
        # For a simple setup, let's use:
        # 1. Current CityFlow simulation step
        # 2. Total number of vehicles in CityFlow
        # 3. Total queue length in CityFlow
        # 4. (Optional) Odom data from one TurtleBot (tb3_0) if you decide to use it.
        
        cityflow_obs_dim = 3 # step, num_vehicles, total_queue_length
        odom_obs_dim = 3 # x, y, yaw (if included for tb3_0)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(cityflow_obs_dim + odom_obs_dim,), 
            dtype=np.float32
        )

        # ACTION SPACE: This defines what your RL agent *does* to the CityFlow environment.
        # Let's assume the RL agent controls a single traffic light in CityFlow for simplicity.
        # We define 2 actions: 0 to set phase 0, 1 to set phase 1 for 'intersection_1'.
        self.action_space = gym.spaces.Discrete(2) # e.g., 2 phases for a traffic light

    def _create_dummy_cityflow_files(self):
            # Define the content for roadnet.json
            roadnet_content = {
                "intersections": [
                    {"id": "intersection_1", "point": {"x": 0.0, "y": 0.0}, "width": 10.0,
                    "roads": ["road_0_1_0", "road_0_1_1"],
                    "roadLinks": [
                        {"type": "go_straight", "startRoad": "road_0_1_0", "endRoad": "road_0_1_1", "direction": 0,
                        "laneLinks": [{"startLaneIndex": 0, "endLaneIndex": 0, "points": [{"x": -5.0, "y": 0.0}, {"x": 5.0, "y": 0.0}]}]}],
                    "trafficLight": {"roadLinkIndices": [0], "lightphases": [{"time": 30, "availableRoadLinks": [0]}, {"time": 30, "availableRoadLinks": []}]},
                    "virtual": False
                    },
                    {"id": "intersection_0", "point": {"x": -100.0, "y": 0.0}, "width": 10.0, "roads": ["road_0_1_0"], "roadLinks": [], "virtual": True},
                    {"id": "intersection_2", "point": {"x": 100.0, "y": 0.0}, "width": 10.0, "roads": ["road_0_1_1"], "roadLinks": [], "virtual": True}
                ],
                "roads": [ # This is the "roads" key CityFlow is complaining about
                    {"id": "road_0_1_0", "startIntersection": "intersection_0", "endIntersection": "intersection_1",
                    "points": [{"x": -100.0, "y": 0.0}, {"x": -5.0, "y": 0.0}], "lanes": [{"width": 3.0, "maxSpeed": 16.67}]},
                    {"id": "road_0_1_1", "startIntersection": "intersection_1", "endIntersection": "intersection_2",
                    "points": [{"x": 5.0, "y": 0.0}, {"x": 100.0, "y": 0.0}], "lanes": [{"width": 3.0, "maxSpeed": 16.67}]}
                ]
            }
            
            # Define the content for flow.json
            flow_content = [
                {"vehicle": [{"startTime": 0, "endTime": -1, "interval": 5.0, "route": ["road_0_1_0", "road_0_1_1"]}]}
            ]
            
            # Define the content for config.json
            config_content = {
                "interval": 1.0, "seed": 0, "dir": "./",
                "roadnetFile": os.path.basename(self.roadnet_path),
                "flowFile": os.path.basename(self.flow_path),
                "rlTrafficLight": True,
                "saveReplay": False, # Changed to False to simplify
                # Removed roadnetLogFile and replayLogFile for simplicity.
            }

            # Write files with error handling
            try:
                with open(self.roadnet_path, 'w') as f:
                    json.dump(roadnet_content, f, indent=2)
                with open(self.flow_path, 'w') as f:
                    json.dump(flow_content, f, indent=2)
                with open(self.config_path, 'w') as f:
                    json.dump(config_content, f, indent=2)
                self.ros_node.get_logger().info("Dummy CityFlow files created for Gym env.")
            except IOError as e:
                self.ros_node.get_logger().error(f"Error writing CityFlow dummy files: {e}")
                raise # Re-raise to stop if file writing fails

    def _get_cityflow_observation(self):
        """Extracts relevant observation data from CityFlow."""
        num_vehicles = len(self.cityflow_engine.get_vehicles())
        total_queue_len = sum(self.cityflow_engine.get_lane_queue_length().values())
        return np.array([float(self.cityflow_step_counter), float(num_vehicles), float(total_queue_len)])

    def _get_odom_observation(self, robot_id='tb3_0'):
        """Extracts x, y, yaw from TurtleBot odometry for observation (if used)."""
        odom = self.current_odom_data.get(robot_id)
        if odom:
            pos = odom.pose.pose.position
            quat = odom.pose.pose.orientation
            # Simple yaw calculation from quaternion (assuming 2D motion)
            yaw = 2 * math.atan2(quat.z, quat.w) 
            return np.array([pos.x, pos.y, yaw])
        return np.zeros(3) # Return zeros if no data for this robot

    def _get_obs(self):
        """Combines CityFlow and (optional) TurtleBot observations."""
        cityflow_obs = self._get_cityflow_observation()
        # Include tb3_0's odom in the observation as an example of "using it"
        odom_obs = self._get_odom_observation('tb3_0') 
        return np.concatenate([cityflow_obs, odom_obs])

    def _get_info(self):
        """Provides supplementary information for debugging/logging."""
        return {
            "cityflow_step": self.cityflow_step_counter,
            "cityflow_vehicles": len(self.cityflow_engine.get_vehicles()),
            "cityflow_total_queue_length": sum(self.cityflow_engine.get_lane_queue_length().values()),
            "tb3_0_odom": self.current_odom_data.get('tb3_0') # Raw Odom message
        }

    def reset(self, seed=None, options=None):
        """
        Resets the environment to an initial state.
        Args:
            seed: An optional seed for reproducibility.
            options: Optional dictionary of options for reset.
        Returns:
            observation (numpy.ndarray): The initial observation.
            info (dict): A dictionary containing supplementary information.
        """
        super().reset(seed=seed) # Important for Gym's internal state
        self.ros_node.get_logger().info("Gym environment reset called. Resetting CityFlow.")
        self.cityflow_engine.reset() # Reset CityFlow simulation
        self.cityflow_step_counter = 0

        # Stop all TurtleBots by publishing zero velocity
        for pub in self.cmd_vel_publishers.values():
            stop_msg = Twist()
            pub.publish(stop_msg)
            
        # IMPORTANT: If your TurtleBots are in Gazebo, you might need a ROS2 service call
        # to reset their positions there for true episode resets.
        # This is beyond the "minimal" scope but crucial for robust RL training.
        # Example: call /gazebo/set_entity_state service

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        """
        Performs one step in the environment.
        Args:
            action: An action from the RL agent (e.g., traffic light phase).
        Returns:
            observation (numpy.ndarray): The observation of the environment.
            reward (float): The reward received.
            done (bool): Whether the episode has ended (True for terminated or truncated).
            info (dict): A dictionary containing supplementary information.
        """
        # 1. Apply action to CityFlow (e.g., change traffic light phase)
        # This uses the action to control 'intersection_1' in CityFlow.
        if self.cityflow_engine.get_traffic_lights().get('intersection_1'):
             self.cityflow_engine.set_traffic_light_phase('intersection_1', int(action))
             self.ros_node.get_logger().info(f"CityFlow: Set TL 'intersection_1' to phase {int(action)}")

        # 2. Advance CityFlow simulation
        self.cityflow_engine.next_step()
        self.cityflow_step_counter += 1

        # 3. Send cmd_vel to TurtleBots (side effect, independent of RL action if desired)
        # Here, we make tb3_0 move forward slowly, and others stay still.
        # This is a fixed demo behavior for TurtleBots. You can make this conditional
        # on CityFlow state (e.g., TurtleBot follows a simulated vehicle) or just random.
        cmd_vel_msg = Twist()
        # Make tb3_0 move forward if the CityFlow step counter is even, stop if odd
        if self.cityflow_step_counter % 2 == 0: 
             cmd_vel_msg.linear.x = 0.1 # Move forward
             cmd_vel_msg.angular.z = 0.0 # No rotation
        else:
             cmd_vel_msg.linear.x = 0.0
             cmd_vel_msg.angular.z = 0.0

        self.cmd_vel_publishers['tb3_0'].publish(cmd_vel_msg) # Only control tb3_0 for simplicity
        # For other turtlebots, they'll remain stopped unless you publish to their cmd_vel topics too.

        # 4. Get new observation (from CityFlow and optional Odom)
        observation = self._get_obs()

        # 5. Calculate reward (based on CityFlow state)
        # Example: penalize total queue length, encourage traffic flow (less vehicles)
        total_queue_len = sum(self.cityflow_engine.get_lane_queue_length().values())
        num_vehicles = len(self.cityflow_engine.get_vehicles())
        reward = -0.05 * total_queue_len - 0.01 * num_vehicles # Penalize queues and too many vehicles

        # 6. Check for episode termination
        # In gym 0.26.2, 'done' covers both terminated and truncated.
        done = self.cityflow_step_counter >= 300 # Stop episode after 300 CityFlow steps
        # No separate 'terminated' or 'truncated' return values for gym 0.26.2

        info = self._get_info()

        # IMPORTANT: To get fresh Odom data, ensure ROS2 messages are processed.
        # The main `rclpy.spin` loop handles subscriber callbacks.
        
        return observation, reward, done, info # 4-tuple return for gym 0.26.2

    def _odom_callback(self, msg, robot_id):
        """Callback to store the latest odometry data for a robot."""
        self.current_odom_data[robot_id] = msg
        # self.ros_node.get_logger().info(f"Received odom for {robot_id}") # Can be verbose

    def cleanup_cityflow_files(self):
        """Removes temporary CityFlow configuration and log files."""
        files_to_remove = [
            self.config_path, self.roadnet_path, self.flow_path,
            "roadnet.log", "replay.log" # CityFlow logs
        ]
        for file in files_to_remove:
            if os.path.exists(file):
                try:
                    os.remove(file)
                    self.ros_node.get_logger().info(f"Cleaned up {file}")
                except OSError as e:
                    self.ros_node.get_logger().warn(f"Could not remove {file}: {e}")
        self.ros_node.get_logger().info("Finished CityFlow file cleanup.")

# --- ROS2 Node Wrapper ---
class RL_CityFlow_Node(Node):
    def __init__(self):
        super().__init__('rl_cityflow_node')
        self.get_logger().info("RL CityFlow Node started, integrating Gym environment.")

        # Initialize the custom Gym environment, passing this ROS2 node instance
        self.env = CityFlowGymEnv(self, num_turtlebots=3) 

        # --- RL Loop Management ---
        self.episode_count = 0
        self.total_reward = 0.0
        self.current_episode_steps = 0
        self.max_episode_steps = 300 # Matches env termination condition for now

        # Set up a ROS2 timer to periodically call the Gym environment's step.
        # This timer drives the entire RL simulation (CityFlow + TurtleBot commands).
        # Match this frequency to your desired CityFlow interval (e.g., 1.0 second per CityFlow step)
        self.timer = self.create_timer(1.0, self.rl_loop_callback) 
        
        # Initial reset of the environment to start the first episode
        self.observation, self.info = self.env.reset() # 2-tuple return for gym 0.26.2 reset
        self.get_logger().info(f"Initial Gym environment reset complete. Observation shape: {self.observation.shape}")

    def rl_loop_callback(self):
        """
        This callback drives the RL training loop.
        It calls env.step(), collects rewards, and handles episode resets.
        """
        # For this simple example, we use a random action.
        # In a real RL setup, you would replace this with your actual policy's action.
        action = self.env.action_space.sample() 
        
        # Take a step in the environment
        # Changed to 4-tuple unpacking for gym 0.26.2
        observation, reward, done, info = self.env.step(action) 
        
        self.total_reward += reward
        self.current_episode_steps += 1

        self.get_logger().info(
            f"RL Step {self.current_episode_steps:03d} | Action={action}, Reward={reward:.2f}, "
            f"Total Reward={self.total_reward:.2f} | CF Vehicles: {info['cityflow_vehicles']}, CF Queue: {info['cityflow_total_queue_length']:.1f}"
        )
        
        # Log tb3_0 odom if available in info (for monitoring)
        if info['tb3_0_odom']:
             odom_pos = info['tb3_0_odom'].pose.pose.position
             self.get_logger().info(f"  TB3_0 Odom: x={odom_pos.x:.2f}, y={odom_pos.y:.2f}, z={odom_pos.position.z:.2f}")

        # Check for episode termination
        if done: # 'done' boolean encompasses termination and truncation in gym 0.26.2
            self.get_logger().info(
                f"--- Episode {self.episode_count} Finished. Total Reward: {self.total_reward:.2f} "
                f"after {self.current_episode_steps} steps. ---"
            )
            self.episode_count += 1
            self.current_episode_steps = 0
            # Reset environment for the next episode
            self.observation, self.info = self.env.reset() 

    def destroy_node(self):
        """Clean up resources before node shutdown."""
        self.env.cleanup_cityflow_files() # Ensure CityFlow temp files are removed
        super().destroy_node()

# --- Main Execution ---
def main(args=None):
    rclpy.init(args=args)
    node = RL_CityFlow_Node()
    try:
        rclpy.spin(node) # Keeps the node alive and processes callbacks
    except KeyboardInterrupt:
        node.get_logger().info("RL Node interrupted. Shutting down.")
    finally:
        node.destroy_node() # Calls the destroy_node method for cleanup
        rclpy.shutdown()

if __name__ == '__main__':
    main()