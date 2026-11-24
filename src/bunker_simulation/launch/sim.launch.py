import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, GroupAction, IncludeLaunchDescription,
                            RegisterEventHandler, SetEnvironmentVariable,
                            TimerAction)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def gpu_env():
    """Rendering με την GPU της NVIDIA μέσα στο WSL2 (Mesa D3D12) αντί για llvmpipe (CPU).

    Χωρίς αυτά το Mesa πέφτει σε software rendering. Απενεργοποίηση: use_gpu:=false.
    """
    if not os.path.exists('/dev/dxg'):      # όχι WSL2: άφησε το σύστημα όπως είναι
        return []
    ld = os.pathsep.join(filter(None, ['/usr/lib/wsl/lib', os.environ.get('LD_LIBRARY_PATH', '')]))
    return [
        SetEnvironmentVariable('GALLIUM_DRIVER', 'd3d12'),
        SetEnvironmentVariable('MESA_D3D12_DEFAULT_ADAPTER_NAME', 'NVIDIA'),
        SetEnvironmentVariable('LD_LIBRARY_PATH', ld),
    ]


def generate_launch_description():
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')
    pkg_bunker_sim = get_package_share_directory('bunker_simulation')

    world_file = os.path.join(pkg_bunker_sim, 'worlds', 'cave_circuit_practice_01.sdf')
    xacro_file = os.path.join(pkg_bunker_sim, 'urdf', 'bunker_mini.urdf.xacro')
    bridge_config = os.path.join(pkg_bunker_sim, 'config', 'bridge.yaml')

    run_open_loop = LaunchConfiguration('run_open_loop')
    headless = LaunchConfiguration('headless')

    declare_args = [
        DeclareLaunchArgument('run_open_loop', default_value='false',
                              description='Start the old open-loop state_manager (drives a fixed '
                                          'sequence; do not combine with autonomy.launch.py)'),
        DeclareLaunchArgument('headless', default_value='false',
                              description='Run Gazebo server only (no GUI)'),
        DeclareLaunchArgument('use_gpu', default_value='true',
                              description='Render with the NVIDIA GPU (WSL2 Mesa D3D12)'),
        DeclareLaunchArgument('x', default_value='0.0'),
        DeclareLaunchArgument('y', default_value='0.0'),
        DeclareLaunchArgument('z', default_value='0.15'),
        DeclareLaunchArgument('yaw', default_value='0.0'),
    ]

    # Ώστε το Gazebo να βρίσκει materials, meshes και worlds του πακέτου
    set_gz_resource_path = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=os.path.dirname(pkg_bunker_sim),
    )

    # 1. Gazebo με τον κόσμο της σπηλιάς (-r: ξεκινά αμέσως η προσομοίωση)
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')),
        launch_arguments={
            # '-s' = server only (χωρίς GUI) όταν headless:=true
            'gz_args': [f'-r -v 3 {world_file}',
                        PythonExpression(["' -s' if '", headless, "' == 'true' else ''"])],
            'on_exit_shutdown': 'true',
        }.items(),
    )

    # 2. robot_description από το xacro -> robot_state_publisher (TF των links)
    robot_description = ParameterValue(Command(['xacro ', xacro_file]), value_type=str)
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': robot_description,
                     'use_sim_time': True}],
        output='screen',
    )

    # 3. Spawn του Bunker Mini μέσα στον κόσμο
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-world', 'cave_circuit_practice_01',
            '-name', 'bunker_mini',
            '-topic', 'robot_description',
            '-x', LaunchConfiguration('x'),
            '-y', LaunchConfiguration('y'),
            '-z', LaunchConfiguration('z'),
            '-Y', LaunchConfiguration('yaw'),
        ],
        output='screen',
    )

    # 4. ROS <-> Gazebo bridge (clock, cmd_vel, odom, tf, joint_states)
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        parameters=[{'config_file': bridge_config, 'use_sim_time': True}],
        output='screen',
    )

    # 4b. Διορθώνει το frame του point cloud της κάμερας (βλ. oak_points_relay.py)
    points_relay = Node(
        package='mine_rover_core',
        executable='oak_points_relay',
        parameters=[{'use_sim_time': True}],
        output='screen',
    )

    # 5. Ο open-loop κώδικας. Ξεκινά αφού γίνει spawn και περάσουν 3 s
    #    ώστε το ρομπότ να "κάτσει" στο έδαφος της σπηλιάς.
    state_manager = Node(
        package='mine_rover_core',
        executable='state_manager',
        parameters=[{'use_sim_time': True}],
        output='screen',
        condition=IfCondition(run_open_loop),
    )
    start_state_manager = RegisterEventHandler(
        OnProcessExit(
            target_action=spawn_robot,
            on_exit=[TimerAction(period=3.0, actions=[state_manager])],
        )
    )

    use_gpu = [GroupAction(gpu_env(), condition=IfCondition(LaunchConfiguration('use_gpu')),
                           scoped=False)]

    return LaunchDescription(declare_args + use_gpu + [
        set_gz_resource_path,
        gazebo,
        bridge,
        points_relay,
        robot_state_publisher,
        spawn_robot,
        start_state_manager,
    ])
