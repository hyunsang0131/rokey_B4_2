"""인식·해석만 (로봇 명령 없음): rs_perception + webcam_watch + intent_node.
  ros2 launch assist_arm perception.launch.py [show_debug:=true] [voice:=true]
관찰 자세 신호가 없으니 RealSense 손 인식(rs_hand)은 쉬고, 웹캠 제스처·콕·음성·의도 해석을 확인하는 용도.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch_ros.actions import Node


def setup(ctx):
    P = ctx.launch_configurations['params_file']
    dbg = {'show_debug': ctx.launch_configurations['show_debug'].lower() == 'true'}
    voice = ctx.launch_configurations['voice'].lower() == 'true'
    return [Node(package='assist_arm', executable='rs_perception', output='screen', parameters=[P, dbg]),
            Node(package='assist_arm', executable='webcam_watch', output='screen', parameters=[P, dbg]),
            Node(package='assist_arm', executable='intent_node', output='screen',
                 parameters=[P, {'enabled': True}] if voice else [P])]


def generate_launch_description():
    cfg = os.path.join(get_package_share_directory('assist_arm'), 'config', 'assist_arm.yaml')
    return LaunchDescription([DeclareLaunchArgument('params_file', default_value=cfg),
                              DeclareLaunchArgument('show_debug', default_value='false'),
                              DeclareLaunchArgument('voice', default_value='false'),
                              OpaqueFunction(function=setup)])
