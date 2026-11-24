"""
Artifact detection for the Science Day demo (run it next to sim.launch.py).

    ros2 launch bunker_simulation detection.launch.py
    ros2 launch bunker_simulation detection.launch.py model:=/path/to/best.pt rviz:=true

Starts:
  artifact_detector  YOLO + depth -> 3D detections (simulation stand-in for the
                     on-camera YOLO of the real OAK-D Lite)
  artifact_reporter  detections -> artifacts on the map + RViz markers
  rviz2              (optional) demo screen: camera with boxes, robot, artifacts
"""
import glob
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def yolo_env():
    """Puts the YOLO virtualenv (~/venvs/yolo, see docs/YOLO_TRAINING.md) on PYTHONPATH."""
    sites = glob.glob(os.path.expanduser('~/venvs/yolo/lib/python3*/site-packages'))
    current = os.environ.get('PYTHONPATH', '')
    if not sites or sites[0] in current.split(os.pathsep):
        return []
    return [SetEnvironmentVariable('PYTHONPATH', os.pathsep.join(filter(None, [current, sites[0]])))]


def generate_launch_description():
    pkg = get_package_share_directory('bunker_simulation')
    model = LaunchConfiguration('model')
    confidence = LaunchConfiguration('confidence')
    fixed_frame = LaunchConfiguration('fixed_frame')

    return LaunchDescription(yolo_env() + [
        DeclareLaunchArgument('model', default_value='yolov8n.pt',
                              description='YOLO weights (.pt). Default: COCO-pretrained YOLOv8n'),
        DeclareLaunchArgument('confidence', default_value='0.4'),
        # COCO has 80 classes; keep only the artifacts. For your own model use e.g.
        # classes:='survivor,backpack,fire_extinguisher,helmet' (or '' = all)
        DeclareLaunchArgument('classes', default_value='person,backpack'),
        DeclareLaunchArgument('fixed_frame', default_value='odom'),
        DeclareLaunchArgument('pose_topic', default_value='',
                              description="Robot pose for placing finds on the map ('' = TF odom)"),
        DeclareLaunchArgument('rviz', default_value='false'),

        Node(
            package='mine_rover_core', executable='artifact_detector',
            parameters=[{'use_sim_time': True,
                         'model': model,
                         'confidence': confidence,
                         'classes': LaunchConfiguration('classes')}],
            output='screen'),

        Node(
            package='mine_rover_core', executable='artifact_reporter',
            parameters=[{'use_sim_time': True,
                         'fixed_frame': fixed_frame,
                         'pose_topic': LaunchConfiguration('pose_topic')}],
            output='screen'),

        Node(
            package='rviz2', executable='rviz2',
            arguments=['-d', os.path.join(pkg, 'config', 'demo.rviz')],
            parameters=[{'use_sim_time': True}],
            condition=IfCondition(LaunchConfiguration('rviz')),
            output='screen'),
    ])
