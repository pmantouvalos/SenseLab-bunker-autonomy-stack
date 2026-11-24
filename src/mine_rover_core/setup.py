from glob import glob

from setuptools import find_packages, setup

package_name = 'mine_rover_core'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/dashboard', glob('dashboard/*.html')),
        ('share/' + package_name + '/models', glob('models/*.pt')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mad-pan',
    maintainer_email='madpan01@gmail.com',
    description='Lane following, artifact detection, state machine and dashboard for the Bunker Mini 2.0 UGV',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'state_manager = mine_rover_core.state_manager:main',
            'artifact_detector = mine_rover_core.artifact_detector:main',
            'artifact_reporter = mine_rover_core.artifact_reporter:main',
            'oak_points_relay = mine_rover_core.oak_points_relay:main',
            'lane_detector = mine_rover_core.lane_detector:main',
            'obstacle_detector = mine_rover_core.obstacle_detector:main',
            'slalom_planner = mine_rover_core.slalom_planner:main',
            'stanley_controller = mine_rover_core.stanley_controller:main',
            'autonomy_manager = mine_rover_core.autonomy_manager:main',
            'dashboard_server = mine_rover_core.dashboard_server:main',
        ],
    },
)
