import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    robot_localization_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[os.path.join(
            get_package_share_directory('rocky_localization'),
            'config', 'ekf.yaml'
        )]
    )

    imu_republisher_node = Node(
        package='rocky_localization',
        executable='imu_republisher',   # ✅ drop the .py extension
        name='imu_republisher_node',
        output='screen'
    )

    slam_toolbox_node = Node(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        parameters=[os.path.join(
            get_package_share_directory('rocky_localization'),
            'config', 'online_async.yaml'
        )]
    )

    return LaunchDescription([
        robot_localization_node,
        imu_republisher_node,
        slam_toolbox_node,
    ])