"""로봇 없이 전체 흐름 확인 (task_manager dry_run — 로그만). 실제 로봇과 섞이지 않게 도메인 분리 권장:
  ROS_DOMAIN_ID=77 ros2 launch assist_arm dry_run.launch.py [tasks:=all] [decision:=true]
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration as LC


def generate_launch_description():
    base = os.path.join(get_package_share_directory('assist_arm'), 'launch', 'bringup_all.launch.py')
    return LaunchDescription([
        DeclareLaunchArgument('tasks', default_value='clear,bring,water,seat'),
        DeclareLaunchArgument('decision', default_value='false'),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(base),
                                 launch_arguments={'dry_run': 'true', 'tasks': LC('tasks'),
                                                   'decision': LC('decision')}.items())])
