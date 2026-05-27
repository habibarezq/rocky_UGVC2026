# from launch import LaunchDescription
# from launch_ros.actions import Node
# from launch.substitutions import LaunchConfiguration
# from launch.actions import DeclareLaunchArgument

# import os
# from ament_index_python.packages import get_package_share_directory

# def generate_launch_description():
#     use_sim_time = LaunchConfiguration('use_sim_time')

#     joy_params = os.path.join(get_package_share_directory('rocky_controller'), 'config', 'joystick.yaml')

#     joy_node = Node(
#         package='joy',
#         executable='joy_node',
#         parameters=[joy_params, {'use_sim_time': use_sim_time}],
#     )

#     teleop_node = Node(
#             package='teleop_twist_joy',
#             executable='teleop_node',
#             name='teleop_node',
#             parameters=[joy_params, {'use_sim_time': use_sim_time}],
#             # CHANGE THIS LINE to use cmd_vel_unstamped:
#             remappings=[('/cmd_vel', '/diff_drive_controller/cmd_vel_unstamped')]
#         )

#     return LaunchDescription([
#         DeclareLaunchArgument(
#             'use_sim_time',
#             default_value='false',
#             description='Use sim time if true'),
#         joy_node,
#         teleop_node,
#     ])

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')

    teleop_keyboard_node = Node(
        package='teleop_twist_keyboard',
        executable='teleop_twist_keyboard',
        name='teleop_twist_keyboard_node',
        output='screen',
        prefix='gnome-terminal --',
        parameters=[{'use_sim_time': use_sim_time}],
    )
    twist_stamper = Node(
    package='rocky_controller',
    executable='twist_stamper',
    name='twist_stamper',
    parameters=[{'use_sim_time': use_sim_time}],
)


    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use sim time if true'),
        teleop_keyboard_node,
        twist_stamper,
    ])