from setuptools import find_packages, setup

package_name = 'assist_yolo'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rokey',
    maintainer_email='rokey@todo.todo',
    description='Configurable YOLO 3D position service for assist_arm',
    license='TODO',
    entry_points={
        'console_scripts': [
            'assist_detection = assist_yolo.assist_detection:main',
        ],
    },
)
