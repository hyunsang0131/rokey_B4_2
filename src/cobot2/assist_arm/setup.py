import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'assist_arm'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'README.md']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'tools'), [f for f in glob('tools/*') if os.path.isfile(f)]),
        (os.path.join('share', package_name, 'models'), [f for f in glob('models/*') if os.path.isfile(f)]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rokey',
    maintainer_email='rokey@todo.todo',
    description='Home assist robot interacting with human hands (no contact)',
    license='TODO',
    entry_points={
        'console_scripts': [
            # 인식층
            'hand_perception = assist_arm.perception.hand_perception:main',      # v2.3 호연: RealSense 손(주) + 웹캠(보조)
            'object_perception = assist_arm.perception.object_perception:main',  # v2.3 은상: 얼룩 (YOLO 는 컨테이너)
            # 예전 이름 (v2.3 전 카메라별 묶음) — 쓰던 명령이 깨지지 않게 남겨 둠. 런치는 위 두 개를 씀
            'rs_perception = assist_arm.perception.rs_perception:main',
            'webcam_watch = assist_arm.perception.webcam_watch:main',
            # 해석층
            'intent_node = assist_arm.intent.intent_node:main',
            # 결정·실행층
            'task_manager = assist_arm.execution.task_manager:main',
            'decision_node = assist_arm.execution.decision_node:main',
            'music_node = assist_arm.execution.music_node:main',
            # 안전층
            'safety_monitor = assist_arm.safety.safety_monitor:main',
        ],
    },
)
