import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    pkg_turtlebot3_gazebo = get_package_share_directory('turtlebot3_gazebo')
    world_file = '/usr/share/gazebo-11/worlds/empty.world'

    # Environment variable for TurtleBot3 model
    os.environ["TURTLEBOT3_MODEL"] = "waffle"

    # Start Gazebo with GUI
    gazebo = ExecuteProcess(
        cmd=['gazebo', '--verbose', world_file, '-s', 'libgazebo_ros_init.so', '-s', 'libgazebo_ros_factory.so', '-s', 'libgazebo_ros_force_system.so'],
        output='screen'
    )

    # Launch description
    launch_description = LaunchDescription()

    # Add Gazebo
    launch_description.add_action(gazebo)

    # Spawn three TurtleBots with different namespaces and positions
    for i in range(3):
        namespace = f'tb3_{i}'
        x_pos = float(i * 2.0)  # Space them out along x-axis
        y_pos = 0.0
        z_pos = 0.01  # Slightly above ground to avoid collision issues
        roll = 0.0
        pitch = 0.0
        yaw = 0.0

        # Spawn TurtleBot3
        spawn_turtlebot = Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=[
                '-entity', f'{namespace}',
                '-file', os.path.join(pkg_turtlebot3_gazebo, 'models', 'turtlebot3_waffle', 'model.sdf'),
                '-x', str(x_pos),
                '-y', str(y_pos),
                '-z', str(z_pos),
                '-R', str(roll),
                '-P', str(pitch),
                '-Y', str(yaw),
                '-robot_namespace', namespace,
                '-spawn_service_timeout', '120'
            ],
            output='screen'
        )

        # Launch TurtleBot3 node with namespace
        turtlebot_node = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_turtlebot3_gazebo, 'launch', 'robot_state_publisher.launch.py')
            ),
            launch_arguments={
                'namespace': namespace,
                'use_sim_time': 'true'
            }.items()
        )

        launch_description.add_action(spawn_turtlebot)
        launch_description.add_action(turtlebot_node)

    return launch_description