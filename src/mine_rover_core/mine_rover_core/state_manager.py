import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist # Velocity message understood by the Bunker

class RoverStateManager(Node):
    def __init__(self):
        super().__init__('state_manager_node')
        self.get_logger().info('Rover Boot Sequence Initiated...')
        self.get_logger().info('Comms: Denied. GPS: Denied. Executing Auto-Return Sequence.')
        
        # 1. Create the publisher that talks to the robot
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # 2. The timer fires 10 times per second (0.1 s) to keep the robot "awake"
        self.timer = self.create_timer(0.1, self.timer_callback)
        
        # 3. State machine variables
        self.state = 0
        # Set in the first callback: with use_sim_time the clock is 0
        # until the first /clock message arrives from Gazebo.
        self.state_start_time = None
        
        # 4. Motion settings (speeds and durations)
        self.forward_speed = 0.5     # m/s
        self.forward_duration = 4.0  # Seconds of motion
        
        self.turn_speed = 0.5        # rad/s
        self.turn_duration = 6.9    # Seconds for 180 degrees (0.5 * 6.28 = 3.14 rad)

    def timer_callback(self):
        msg = Twist()

        if self.state_start_time is None:
            now = self.get_clock().now()
            if now.nanoseconds == 0:
                return  # no simulation clock yet
            self.state_start_time = now

        # How much time has passed since entering the current state?
        elapsed_time = (self.get_clock().now() - self.state_start_time).nanoseconds / 1e9
        
        if self.state == 0:
            # STATE 0: Move forward
            if elapsed_time < self.forward_duration:
                msg.linear.x = self.forward_speed
                self.get_logger().info('State 0: Moving Forward (0.5 m/s)...', once=True)
            else:
                self.transition_to_next_state()
                
        elif self.state == 1:
            # STATE 1: Turn 180 degrees
            if elapsed_time < self.turn_duration:
                msg.angular.z = self.turn_speed
                self.get_logger().info('State 1: Turning 180 degrees...', once=True)
            else:
                self.transition_to_next_state()
                
        elif self.state == 2:
            # STATE 2: Return
            if elapsed_time < self.forward_duration:
                msg.linear.x = self.forward_speed
                self.get_logger().info('State 2: Returning to base...', once=True)
            else:
                self.transition_to_next_state()
                
        elif self.state == 3:
            # STATE 3: Turn 180 degrees (restore heading)
            if elapsed_time < self.turn_duration:
                msg.angular.z = self.turn_speed
                self.get_logger().info('State 3: Re-aligning to original heading...', once=True)
            else:
                self.transition_to_next_state()
                
        elif self.state == 4:
            # STATE 4: Done (park)
            msg.linear.x = 0.0
            msg.angular.z = 0.0
            self.get_logger().info('Sequence Complete. Net displacement: 0.', once=True)
            # Stay here sending zeros so the robot stays still

        # Send the message to the CAN bus
        self.publisher_.publish(msg)

    def transition_to_next_state(self):
        """Helper function for changing state"""
        self.state += 1
        self.state_start_time = self.get_clock().now()
        # Add a short brake between motions so the motors do not jerk
        msg = Twist()
        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = RoverStateManager()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # On Ctrl+C, send a stop command for safety before shutting down
        stop_msg = Twist()
        node.publisher_.publish(stop_msg)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()