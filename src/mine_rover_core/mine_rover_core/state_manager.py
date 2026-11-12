import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist # Το μήνυμα ταχύτητας που καταλαβαίνει το Bunker

class RoverStateManager(Node):
    def __init__(self):
        super().__init__('state_manager_node')
        self.get_logger().info('Rover Boot Sequence Initiated...')
        self.get_logger().info('Comms: Denied. GPS: Denied. Executing Auto-Return Sequence.')
        
        # 1. Δημιουργούμε τον Publisher που μιλάει στο ρομπότ
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # 2. Το timer χτυπάει 10 φορές το δευτερόλεπτο (0.1s) για να κρατάει το ρομπότ "ξύπνιο"
        self.timer = self.create_timer(0.1, self.timer_callback)
        
        # 3. Μεταβλητές της Μηχανής Καταστάσεων
        self.state = 0
        self.state_start_time = self.get_clock().now()
        
        # 4. Ρυθμίσεις κίνησης (Ταχύτητες και Χρόνοι)
        self.forward_speed = 0.5     # m/s
        self.forward_duration = 4.0  # Δευτερόλεπτα κίνησης
        
        self.turn_speed = 0.5        # rad/s
        self.turn_duration = 6.9    # Δευτερόλεπτα για 180 μοίρες (0.5 * 6.28 = 3.14 rad)

    def timer_callback(self):
        msg = Twist()
        
        # Πόσος χρόνος έχει περάσει από τότε που μπήκαμε στην τρέχουσα κατάσταση;
        elapsed_time = (self.get_clock().now() - self.state_start_time).nanoseconds / 1e9
        
        if self.state == 0:
            # ΚΑΤΑΣΤΑΣΗ 0: Κίνηση Μπροστά
            if elapsed_time < self.forward_duration:
                msg.linear.x = self.forward_speed
                self.get_logger().info('State 0: Moving Forward (0.5 m/s)...', once=True)
            else:
                self.transition_to_next_state()
                
        elif self.state == 1:
            # ΚΑΤΑΣΤΑΣΗ 1: Στροφή 180 μοίρες
            if elapsed_time < self.turn_duration:
                msg.angular.z = self.turn_speed
                self.get_logger().info('State 1: Turning 180 degrees...', once=True)
            else:
                self.transition_to_next_state()
                
        elif self.state == 2:
            # ΚΑΤΑΣΤΑΣΗ 2: Επιστροφή
            if elapsed_time < self.forward_duration:
                msg.linear.x = self.forward_speed
                self.get_logger().info('State 2: Returning to base...', once=True)
            else:
                self.transition_to_next_state()
                
        elif self.state == 3:
            # ΚΑΤΑΣΤΑΣΗ 3: Στροφή 180 μοίρες (Επαναφορά Κατεύθυνσης)
            if elapsed_time < self.turn_duration:
                msg.angular.z = self.turn_speed
                self.get_logger().info('State 3: Re-aligning to original heading...', once=True)
            else:
                self.transition_to_next_state()
                
        elif self.state == 4:
            # ΚΑΤΑΣΤΑΣΗ 4: Τέλος (Στάθμευση)
            msg.linear.x = 0.0
            msg.angular.z = 0.0
            self.get_logger().info('Sequence Complete. Net displacement: 0.', once=True)
            # Παραμένουμε εδώ στέλνοντας μηδενικά για να μείνει ακίνητο

        # Στέλνουμε το μήνυμα στο CAN bus
        self.publisher_.publish(msg)

    def transition_to_next_state(self):
        """Βοηθητική συνάρτηση για την αλλαγή κατάστασης"""
        self.state += 1
        self.state_start_time = self.get_clock().now()
        # Βάζουμε ένα μικρό φρένο μεταξύ των κινήσεων για να μην κλωτσήσουν τα μοτέρ
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
        # Όταν πατήσεις Ctrl+C, στέλνει εντολή στάσης για ασφάλεια πριν κλείσει
        stop_msg = Twist()
        node.publisher_.publish(stop_msg)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()