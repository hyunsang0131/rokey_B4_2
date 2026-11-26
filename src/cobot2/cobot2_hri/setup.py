import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'cobot2_hri'
setup(
    name=package_name,
    version='1.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'README.md']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'tools'), [f for f in glob('tools/*') if os.path.isfile(f)]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hoyeon',
    maintainer_email='rokey@todo.todo',
    description='cobot2 HRI: gesture_node, pour_level_node, pour_module',
    license='TODO',
    entry_points={
        'console_scripts': [
            'gesture_node = cobot2_hri.gesture_node:main',
            'pour_level_node = cobot2_hri.pour_level_node:main',
            'fake_skill_server = cobot2_hri.fake_skill_server:main',
            'fake_task_manager = cobot2_hri.fake_task_manager:main',
        ],
    },
)
