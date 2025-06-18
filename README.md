# MultiRobotNavigationCityFlow
This is the Repository for the Open project of the course (DTEK2084) Aerial Robotics and Multi-Robot Systems 2025 at the University of Turku. Team Members Ameer Moavia, Ayana Guruge, Syed Mehdi 
# Setup
The Following needs to be installed as a basic requirement to run the project.
1. Linux 22.04 LTS jammy jellyfish
2. ROS2 Humble Hawksbill
3. Python = 3.10
4. Cityflow v0.1 from the Git
5. OpenAI gym 0.26.2
6. stablebaselines3
   and all their dependencies.
# How to run the simulation and create a model
1.define your configfile,flow and roadnet in /cityflow_rl/cityflow_config. 
2.then run train_city_flow.py inside the /cityflow_rl directory.
3. model wil be created
# Visualizing cityflow
1. In order to run the replay simulation of the model, go to CityflowInstallDirectory/frontend and run index.html
2. It will open in a browser. here you can input replay_roadnet.json and replay.txt from cityflow_rl/cityflow_config
3. run to visualize.
# Running Turtlebot3 Gazebo Environment and link
1. source ros2 and gazebo properly
   source /opt/ros/humble/setup.bash
   source /usr/share/gazebo/setup.sh
2. colcon build and source the install
3. go to /turtlebot3_ws
4. run the following
   ros2 launch multi_tb3_sim multi_turtlebot3.launch.py
