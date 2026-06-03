from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    package_share = get_package_share_directory("person_follower")
    config_file = os.path.join(package_share, "config", "config.yaml")

    return LaunchDescription(
        [
            Node(
                package="person_follower",
                executable="person_follower_node",
                name="person_follower",
                output="screen",
                parameters=[config_file],
            )
        ]
    )
