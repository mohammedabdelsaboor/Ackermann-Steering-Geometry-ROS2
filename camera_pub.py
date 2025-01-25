#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class CameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_pub_node')

        self.publisher_ = self.create_publisher(Image, '/camera/image_raw', 10)
        self.timer = self.create_timer(0.1, self.publish_image) 
        self.bridge = CvBridge() 
        self.cap = cv2.VideoCapture(0) 

        if not self.cap.isOpened():
            self.get_logger().error("Unable to open the camera!")
            rclpy.shutdown()

    def publish_image(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error("Failed to capture image!")
            return

        image_message = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')

        self.publisher_.publish(image_message)
        self.get_logger().info('Published image')

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()


def main(args=None):
    try:
        rclpy.init(args=args)
    except Exception as e:
        print(f"Failed to initialize rclpy: {e}")
        return

    node = CameraPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
