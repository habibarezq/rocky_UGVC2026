#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import time
from sensor_msgs.msg import Imu

imu_pub = None

def ImuCallback(imu):
    global imu_pub
    imu.header.frame_id = "base_footprint_ekf"
    imu_pub.publish(imu)



def main():
    global imu_pub
    rclpy.init()
    node = Node("imu_republisher")
    time.sleep(1)  # Wait for other nodes to initialize
    imu_pub = node.create_publisher(
        Imu, "imu/ekf", 10
    )
    imu_sub = node.create_subscription(
        Imu , "/imu", ImuCallback , 10   
    )
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    



if __name__ == '__main__':
    main()    