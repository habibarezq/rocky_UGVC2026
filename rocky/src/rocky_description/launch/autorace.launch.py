import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, RegisterEventHandler, TimerAction
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():

    package_name = 'rocky_description'

    pkg_share = get_package_share_directory(package_name)
    world_file = os.path.join(pkg_share, 'worlds', 'autorace.world')

    rsp = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(pkg_share, 'launch', 'rsp.launch.py')]),
        launch_arguments={'use_sim_time': 'true', 'use_ros2_control': 'true'}.items()
    )

    # ✅ Use 'gz_args' not 'world', and pass -r to auto-run simulation
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')]),
        launch_arguments={
            'gz_args': '-r ' + world_file,
            'on_exit_shutdown': 'true'
        }.items()
    )

    # ✅ ROS-GZ bridge is required — gz_sim.launch.py does NOT start it
    ros_gz_bridge = Node(
    package='ros_gz_bridge',
    executable='parameter_bridge',
     arguments=[
        '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
        '/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',  # ADD THIS
        '/imu@sensor_msgs/msg/Imu[gz.msgs.IMU',               # ADD THIS too
    ],
    parameters=[{'use_sim_time': True}],  # ADD THIS
    output='screen'
)


    spawn_entity = Node(
    package='ros_gz_sim',
    executable='create',
    arguments=[
        '-topic', 'robot_description',
        '-name', 'rocky',
        '-x', '0.4',
        '-y', '-1.75',
        '-z', '0.05',
        '-Y', '1.5707'
    ]
)

    delayed_spawn = TimerAction(period=8.0, actions=[spawn_entity])
    delayed_rsp = TimerAction(period=2.0, actions=[rsp])

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
        gazebo,
        ros_gz_bridge,   # bridge must be running before spawn
        delayed_rsp,
        delayed_spawn,
        delay_joint_state_broadcaster_after_spawn,
        delay_diff_drive_spawner_after_joint_state,
    ])