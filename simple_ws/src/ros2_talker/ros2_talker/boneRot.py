import rclpy
from rclpy.node import Node
import numpy as np
from std_msgs.msg import Float32MultiArray

class BonePublisher(Node):

    def __init__(self):
        super().__init__('bone_publisher')
        self.publisher_ = self.create_publisher(Float32MultiArray, 'bone_rotations', 10)
        self.timer_period = 0.1
        self.timer = self.create_timer(self.timer_period, self.timer_callback)
        self.num_bones = 15
        self.time_counter = 0.0
        self.oscillation_speed = 12.0 
        self.rotation_limit = 8.0 

    def timer_callback(self):

        bone_rotations = Float32MultiArray()
        self.time_counter += self.timer_period * self.oscillation_speed 

        bone_rotations.data = [
            np.sin(self.time_counter + i * 0.6) * self.rotation_limit # Sumar 3 para girar a la derecha, restar 3 para la izquierda, no sumar nada para avanzar.
            for i in range(self.num_bones)
        ]

        self.publisher_.publish(bone_rotations)
        self.get_logger().info(f'Publishing bone rotations: {bone_rotations.data}')

def main(args=None):
    rclpy.init(args=args)
    bone_publisher = BonePublisher()
    rclpy.spin(bone_publisher)

    # Cleanup
    bone_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
