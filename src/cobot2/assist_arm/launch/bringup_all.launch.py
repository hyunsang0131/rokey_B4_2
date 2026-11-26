"""전체 실행 (4층 구조). 로봇이 움직여요 — dry_run:=true 면 로그만.

  ros2 launch assist_arm bringup_all.launch.py                              # P0+P1 (치우기·가져오기·물·자리 세팅)
  ros2 launch assist_arm bringup_all.launch.py tasks:=clear,bring           # P0 만
  ros2 launch assist_arm bringup_all.launch.py tasks:=all decision:=true    # 전부 + 먼저 제안
  ros2 launch assist_arm bringup_all.launch.py voice:=true                  # 음성 명령 (get_keyword 와 동시 사용 금지)
tasks 이름: clear(치우기 3종) bring(가져오기) water(물) seat(자리 세팅) music(음악·율동) wipe(테이블 닦기) / all
사전 실행: sodreal(두산), realsense(카메라), 컨테이너 assist_detection(YOLO)
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch_ros.actions import Node

TASK_GROUPS = {'clear': ['clear_dish', 'clear_empty', 'clear_all'], 'bring': ['bring_dish'],
               'water': ['pour_water'], 'seat': ['seat_setting'], 'music': ['music'], 'wipe': ['wipe_table']}
ALL_TASKS = [t for g in TASK_GROUPS.values() for t in g]
ALWAYS = ['observe_table', 'scan_scene', 'express']      # 관찰·스냅샷·제안 모션은 항상


def _b(ctx, name):
    return ctx.launch_configurations[name].lower() in ('1', 'true', 'yes', 'on')


def setup(ctx):
    P = ctx.launch_configurations['params_file']
    sel = [x.strip() for x in ctx.launch_configurations['tasks'].split(',') if x.strip()]
    enabled = set(ALL_TASKS) if 'all' in sel else {t for g in sel for t in TASK_GROUPS.get(g, [])}
    unknown = [g for g in sel if g != 'all' and g not in TASK_GROUPS]
    if unknown:
        print(f'[bringup_all] 모르는 tasks: {unknown} (가능: {list(TASK_GROUPS)} / all)')
    tm = {'dry_run': _b(ctx, 'dry_run'), **{f'enable_{t}': (t in enabled or t in ALWAYS) for t in ALL_TASKS + ALWAYS}}
    dbg = {'show_debug': _b(ctx, 'show_debug')}
    nodes = [
        Node(package='assist_arm', executable='hand_perception', output='screen', parameters=[P, dbg]),
        Node(package='assist_arm', executable='object_perception', output='screen', parameters=[P, dbg]),
        Node(package='assist_arm', executable='intent_node', output='screen',
             parameters=[P, {'enabled': _b(ctx, 'voice')}] if _b(ctx, 'voice') else [P]),
        Node(package='assist_arm', executable='safety_monitor', output='screen', parameters=[P]),
        Node(package='assist_arm', executable='task_manager', output='screen', parameters=[P, tm]),
    ]
    if _b(ctx, 'decision'):
        nodes.append(Node(package='assist_arm', executable='decision_node', output='screen', parameters=[P]))
    if 'music' in enabled:
        nodes.append(Node(package='assist_arm', executable='music_node', output='screen', parameters=[P]))
    return nodes


def generate_launch_description():
    cfg = os.path.join(get_package_share_directory('assist_arm'), 'config', 'assist_arm.yaml')
    args = {'params_file': cfg, 'tasks': 'clear,bring,water,seat', 'dry_run': 'false', 'decision': 'false',
            'voice': 'false', 'show_debug': 'false'}
    return LaunchDescription([DeclareLaunchArgument(k, default_value=v) for k, v in args.items()]
                             + [OpaqueFunction(function=setup)])
