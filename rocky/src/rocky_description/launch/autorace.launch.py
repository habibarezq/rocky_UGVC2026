import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():

    # !!! MAKE SURE YOU SET THE PACKAGE NAME CORRECTLY !!!
    package_name = 'rocky_description' 

    # Paths
    pkg_share = get_package_share_directory(package_name)
    world_file = os.path.join(pkg_share, 'worlds', 'autorace.world')
    gazebo_params_file = os.path.join(pkg_share, 'config', 'gazebo_params.yaml')

    # Include the robot_state_publisher
    rsp = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(pkg_share, 'launch', 'rsp.launch.py')]),
        launch_arguments={'use_sim_time': 'true', 'use_ros2_control': 'true'}.items()
    )

    # Include Gazebo
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')]),
        launch_arguments={
            'world': world_file,
            'extra_gazebo_args': '--ros-args --params-file ' + gazebo_params_file
        }.items()
    )

    # Spawn Entity in Gazebo
    spawn_entity = Node(
        package='gazebo_ros', 
        executable='spawn_entity.py',
        arguments=['-topic', 'robot_description',
                   '-entity', 'rocky',
                   '-x', '0.0',
                   '-y', '0.0',
                   '-z', '0.05', 
                   '-Y', '0.0'],
        output='screen'
    )

    # Note: These names MUST match the names in your my_controllers.yaml exactly.
    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster"],
    )

    diff_drive_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["diff_drive_controller"],
    )

    # Delay the controllers until Gazebo finishes spawning the robot
    delay_joint_state_broadcaster_after_spawn = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=spawn_entity,
            on_exit=[joint_state_broadcaster_spawner],
        )
    )

    delay_diff_drive_spawner_after_joint_state = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=joint_state_broadcaster_spawner,
            on_exit=[diff_drive_spawner],
        )
    )



    return LaunchDescription([
        rsp,
        gazebo,
        spawn_entity,
        delay_joint_state_broadcaster_after_spawn,
        delay_diff_drive_spawner_after_joint_state,
        
    ])