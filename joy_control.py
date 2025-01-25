import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy

class JoystickController(Node):
    def __init__(self):
        super().__init__('joystick_controller')
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(Joy, '/joy', self.joy_callback, 10)
        self.linear_axis = 1 
        self.angular_axis = 0  
        self.angular_y_axis = 2
        self.linear_y_axis = 3

        self.dead_zone = 0.1 

        self.get_logger().info('Joystick Controller Node Initialized.')

    def scale_to_range(self, value):
        """
        Scale joystick input from [-1, 1] to [1, -1].
        If within the dead zone, return 0.
        """
        if abs(value) < self.dead_zone:
            return 0.0  
        return value

    def joy_callback(self, msg: Joy):
        twist = Twist()
        twist.linear.x = self.scale_to_range(msg.axes[self.linear_axis])
        twist.angular.z = self.scale_to_range(msg.axes[self.angular_axis])
        twist.angular.y = self.scale_to_range(msg.axes[self.angular_y_axis])
        twist.linear.y = self.scale_to_range(msg.axes[self.linear_y_axis])

        self.publisher.publish(twist)
        self.get_logger().info(
            f'Publishing cmd_vel: linear.x={twist.linear.x:.2f}, angular.z={twist.angular.z:.2f} , angular.y={twist.angular.y:.2f},linear.y={twist.linear.y:.2f}'
        )

def main(args=None):
    rclpy.init(args=args)

    joystick_controller = JoystickController()

    rclpy.spin(joystick_controller)

    
    joystick_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()