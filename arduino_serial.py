import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import serial

class CmdVelToSerial(Node):
    def __init__(self):
        super().__init__('cmd_vel_to_serial')

        self.subscription = self.create_subscription(
            Twist,
            'cmd_vel',
            self.cmd_vel_callback,
            10
        )
        self.subscription 
        try:
            self.serial_port = serial.Serial(
                port='/dev/ttyACM0',  
                baudrate=9600,
                timeout=1
            )
            self.get_logger().info('Serial port opened successfully')
        except serial.SerialException as e:
            self.get_logger().error(f'Failed to open serial port: {e}')
            self.serial_port = None

    def cmd_vel_callback(self, msg):
        if self.serial_port and self.serial_port.is_open:
            linear_x = msg.linear.x
            angular_z = msg.angular.z
            angular_y = msg.angular.y 
            linear_y = msg.linear.y 

        
            
            command = f'{linear_x:.2f},{angular_z:.2f},{angular_y:.2f}\n,{linear_y:.2f}\n'

            try:
                self.serial_port.write(command.encode())
                self.get_logger().info(f'Sent to serial: {command.strip()}')
            except serial.SerialException as e:
                self.get_logger().error(f'Failed to write to serial port: {e}')
        else:
            self.get_logger().warn('Serial port is not open')

    def destroy_node(self):
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
            self.get_logger().info('Serial port closed')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    node = CmdVelToSerial()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Node stopped by user')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
