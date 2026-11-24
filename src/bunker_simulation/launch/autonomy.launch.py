"""
The complete autonomous-rover demo: one command starts everything.

    ros2 launch bunker_simulation autonomy.launch.py
    then open  http://localhost:8080  in a browser

Starts
  sim.launch.py          Gazebo: cave + painted lane + artifacts, Bunker Mini, OAK-D Lite
  detection.launch.py    artifact_detector (YOLO: person = the survivor) + artifact_reporter
  lane_detector          neural network that finds the white lane lines
  obstacle_detector      depth image -> big obstacles in metres (/obstacles)
  slalom_planner         remembers rocks, draws the curve around them (/avoid/path)
  stanley_controller x2  one follows the lane, 'stanley_avoid' follows the slalom curve
  (the first stanley_controller steers along the lane centre)
  autonomy_manager       state machine: MANUAL / AUTO / E-STOP, the only writer of /cmd_vel
  dashboard_server       web dashboard (buttons, cameras, warnings)

Arguments
  headless:=true     no Gazebo window (the dashboard shows the cameras anyway)
  rviz:=true         also open RViz
  model:=...         YOLO weights (default yolov8n.pt, COCO)
  classes:=...       artifacts to report, e.g. 'person,backpack,fire_extinguisher'
  lane_model:=...    lane network weights (default lane_seg.pt installed by mine_rover_core)
  artifact_hold_s:=3 stop for 3 s at every new artifact (0 = keep driving)
"""
import glob
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            SetEnvironmentVariable)
from launch.launch_description_sources import PythonLaunchDescriptionSource
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
    sim_time = {'use_sim_time': True}

    args = [
        DeclareLaunchArgument('headless', default_value='false'),
        DeclareLaunchArgument('rviz', default_value='false'),
        # the survivor (Rescue Randy) is a COCO `person`: the pretrained YOLOv8n is enough
        DeclareLaunchArgument('model', default_value='yolov8n.pt'),
        DeclareLaunchArgument('classes', default_value='person'),
        DeclareLaunchArgument('lane_model', default_value='lane_seg.pt'),
        DeclareLaunchArgument('artifact_hold_s', default_value='0.0'),
        DeclareLaunchArgument('port', default_value='8080'),
    ]

    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg, 'launch', 'sim.launch.py')),
        launch_arguments={'headless': LaunchConfiguration('headless'),
                          'run_open_loop': 'false'}.items())

    detection = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg, 'launch', 'detection.launch.py')),
        launch_arguments={'model': LaunchConfiguration('model'),
                          'classes': LaunchConfiguration('classes'),
                          'pose_topic': '/ground_truth/odom',   # simulator only, see slalom_planner
                          'rviz': LaunchConfiguration('rviz')}.items())

    lane = Node(package='mine_rover_core', executable='lane_detector', output='screen',
                parameters=[sim_time, {'model': LaunchConfiguration('lane_model')}])
    obstacles = Node(package='mine_rover_core', executable='obstacle_detector', output='screen',
                     parameters=[sim_time])
    planner = Node(package='mine_rover_core', executable='slalom_planner', output='screen',
                   parameters=[sim_time, {'pose_topic': '/ground_truth/odom'}])
    stanley_avoid = Node(package='mine_rover_core', executable='stanley_controller',
                         name='stanley_avoid', output='screen',
                         parameters=[sim_time, {'path_topic': '/avoid/path',
                                                'cmd_topic': '/avoid/cmd_vel',
                                                'debug_topic': '/avoid/stanley_debug',
                                                'v_max': 0.3, 'v_min': 0.15}])
    stanley = Node(package='mine_rover_core', executable='stanley_controller', output='screen',
                   parameters=[sim_time])
    manager = Node(package='mine_rover_core', executable='autonomy_manager', output='screen',
                   parameters=[sim_time, {'artifact_hold_s': LaunchConfiguration('artifact_hold_s')}])
    dashboard = Node(package='mine_rover_core', executable='dashboard_server', output='screen',
                     parameters=[sim_time, {'port': LaunchConfiguration('port'),
                                            'pose_topic': '/ground_truth/odom'}])

    return LaunchDescription(yolo_env() + args + [sim, detection, lane, obstacles, planner, stanley_avoid, stanley, manager, dashboard])
