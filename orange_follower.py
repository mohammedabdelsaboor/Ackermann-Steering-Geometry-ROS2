import rclpy
import cv2
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import Float64 
import math


class OrangeFollower(Node):
    def __init__(self):
        super().__init__('orange_follower_node')
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)  
        self.camera_tilt_publisher = self.create_publisher(Float64, '/camera_tilt', 10) 
        self.bridge = CvBridge()
        self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        self.desired_distance = 0.5  
        self.focal_length = 600  
        self.known_width = 15  
        self.frame_center_y = 240  

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        orange_center, distance = self.detect_orange(cv_image)

        if orange_center is not None:
            linear_velocity = self.calculate_linear_velocity(distance)
            angular_velocity = self.calculate_angular_velocity(orange_center)
            tilt_adjustment = self.calculate_tilt_adjustment(orange_center[1])
            self.send_velocity_command(linear_velocity, angular_velocity)
            self.send_tilt_command(tilt_adjustment)
        else:
            self.stop_robot()

    def detect_orange(self, cv_image):
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        lower_orange = np.array([5, 150, 150])
        upper_orange = np.array([15, 255, 255])
        mask = cv2.inRange(hsv, lower_orange, upper_orange)
        masked_image = cv2.bitwise_and(cv_image, cv_image, mask=mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return None, None
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        orange_center = (x + w // 2, y + h // 2)
        distance = self.estimate_distance(w)
        cv2.rectangle(cv_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(cv_image, orange_center, 5, (0, 0, 255), -1)
        cv2.imshow("Orange Detection", cv_image)
        cv2.waitKey(1)
        return orange_center, distance
    def estimate_distance(self, width_in_pixels):
        if width_in_pixels == 0:
            return float('inf')
        distance = (self.known_width * self.focal_length) / width_in_pixels
        return distance / 100.0  # Convert from cm to meters
    def calculate_linear_velocity(self, distance):
        if distance > self.desired_distance:
            return 0.2  # Move forward (adjust speed as needed)
        elif distance < self.desired_distance:
            return -0.2  # Move backward (adjust speed as needed)
        else:
            return 0.0  # Stop if at desired distance
    def calculate_angular_velocity(self, orange_center):
        frame_center_x = 320  # Assuming a 640x480 image
        offset_x = orange_center[0] - frame_center_x
        angular_velocity = 0.0
        if abs(offset_x) > 10:  
            angular_velocity = -float(offset_x) / 100.0  
        return angular_velocity

    def calculate_tilt_adjustment(self, y_position):
        offset_y = y_position - self.frame_center_y
        tilt_adjustment = -float(offset_y) / 100.0  
        self.get_logger().info(f"Tilt adjustment calculated: {tilt_adjustment}")
        return tilt_adjustment

    def send_velocity_command(self, linear_velocity, angular_velocity):
        twist = Twist()
        twist.linear.x = linear_velocity
        twist.angular.z = angular_velocity
        self.publisher.publish(twist)

    def send_tilt_command(self, tilt_adjustment):
        tilt_command = Float64()
        tilt_command.data = tilt_adjustment
        self.get_logger().info(f"Publishing tilt command: {tilt_command.data}")
        self.camera_tilt_publisher.publish(tilt_command)
    def stop_robot(self):
        twist = Twist()
        self.publisher.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    orange_follower = OrangeFollower()
    rclpy.spin(orange_follower)
    orange_follower.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
