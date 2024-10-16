#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np

class PositionPublisher(Node):
    def __init__(self):
        super().__init__('position_publisher')
        self.publisher_x = self.create_publisher(Float32MultiArray, 'bone_rotations', 20)
        self.timer = self.create_timer(0.05, self.publish_positions)  # Publish every 0.1 seconds
        
        self.subscriber_unity_coordinates = self.create_subscription(
            Float32MultiArray,
            'unity_coordinates',
            self.listener_callback_unity_coordinates,
            10
        )
        
        self.subscriber_head_pos = self.create_subscription(
            Float32MultiArray,
            'head_pose',
            self.listener_callback_get_pos,
            10
        )
        
        # Initial setup for the simulation state
        self.head_pose = np.zeros(6)
        self.unity_coordinates = []
        #self.Lamprey = np.array([250, 0, 0, 0, 65, 0])   # Position from Unity
        self.modules = np.zeros((6, 24))
        self.ganglios = np.zeros(15)
        self.out = np.zeros((3, 2))
        self.spine = np.zeros((15, 7))
        self.spine[0, 1] = 1
        #self.Estimulos = np.array([[0, 250, 100000, 0]])  # Stimulus data from Unity
        self.colors = ['b', 'r', 'g']
        self.labels = ['Presa', 'Depredador', 'Obstáculo']
        self.Y = 0; self.C = 0; self.M = 0; self.W = 0; self.K = 0

    def gausiana(self, theta, omega):
        return np.exp((np.vdot(theta, omega)-1)/(2*(0.06**2)))

    def naka(self, u, sigma):
        return (max(u, 0)**2) / (sigma**2 + max(u, 0)**2)

    def retina_360(self, x, y, lamprey, iden):
        retina = np.zeros((3, 24))
        DirectionsL = (np.linspace(0, 165.6, 12)) * np.pi / 180  # Preferred directions
        DirectionsR = (np.linspace(-14.4, -180, 12)) * np.pi / 180  # Preferred directions
        OmL = np.vstack((np.cos(DirectionsL), np.sin(DirectionsL)))
        OmR = np.vstack((np.cos(DirectionsR), np.sin(DirectionsR)))
        th = np.zeros(len(x))
        rho = np.zeros(len(x))
        theta = np.zeros((2, len(x)))

        for i in range(len(x)):
            th[i] = np.arctan2((y[i] - lamprey[1]), (x[i] - lamprey[0])) * 180 / np.pi - lamprey[4]
            theta[:, i] = np.vstack((np.cos(th[i] * np.pi / 180), np.sin(th[i] * np.pi / 180))).flatten()
            rho[i] = np.sqrt((x[i] - lamprey[0])**2 + (y[i] - lamprey[1])**2)

        for q in range(len(x)):
            for i in range(12):
                retina[0, i] += self.gausiana(theta[:, q], OmL[:, i]) * 4 * (iden[q] == 0)
                retina[0, i + 12] += self.gausiana(theta[:, q], OmR[:, i]) * 4 * (iden[q] == 0)
                retina[1, i] += self.gausiana(theta[:, q], OmL[:, i]) * 5 * (rho[q] < 200) * (iden[q] == 1)
                retina[1, i + 12] += self.gausiana(theta[:, q], OmR[:, i]) * 5 * (rho[q] < 200) * (iden[q] == 1)
                retina[2, i] += self.gausiana(theta[:, q], OmL[:, i]) * 3 * (rho[q] < 100) * (iden[q] == 2)
                retina[2, i + 12] += self.gausiana(theta[:, q], OmR[:, i]) * 3 * (rho[q] < 100) * (iden[q] == 2)

        return retina, th, rho

    def sensoriomotor(self, estimulo, module, last, GPi):
        W = [np.eye(12), np.fliplr(np.eye(12)), np.fliplr(np.eye(12))]
        weights_r_r = -np.ones((12, 12)) + np.eye(12)
        weights_aux = np.tril(np.ones((12, 12)))
        tau = 5
        Response = np.zeros((3, 24))
        Auxiliary = np.zeros((3, 24))
        Output = np.zeros((3, 2))

        for m in range(3):
            Response[m, :] = np.clip(
                np.hstack([module[m, :12] + (1 / tau) * (-module[m, :12] + np.dot(W[m], np.flip(estimulo[m, :12])) + weights_r_r @ module[m, :12]),
                           module[m, 12:] + (1 / tau) * (-module[m, 12:] + np.dot(W[m], np.flip(estimulo[m, 12:])) + weights_r_r @ module[m, 12:])]),
                0.0, None)

            Auxiliary[m, :] = np.clip(
                np.hstack([module[3 + m, :12] + (1 / tau) * (-module[3 + m, :12] + np.dot(weights_aux, module[m, :12]) - GPi[m + 3]),
                           module[3 + m, 12:] + (1 / tau) * (-module[3 + m, 12:] + np.dot(weights_aux, module[m, 12:]) - GPi[m + 3])]),
                0.0, None)

            Output[m, :] = np.hstack([last[m, 0] + (1 / 5) * (-last[m, 0] + np.sum(module[3 + m, :12])),
                                      last[m, 1] + (1 / 5) * (-last[m, 1] + np.sum(module[3 + m, 12:]))])

        return np.vstack([Response, Auxiliary]), Output

    def arbitramiento(self, retina, response, Ganglia, distances):
        weights_Gan = np.ones((3, 3)) - np.eye(3)
        stimSTR = retina
        condSTR = np.array([self.Y, self.C * self.M, self.W * self.K])
        TaoGPi = 5
        TaoGPe = 10
        TaoSubTN = 10
        TaoStR = 5

        GPi = np.zeros(3)
        GPe = np.zeros(3)
        SubTN = np.zeros(3)
        StR = np.zeros(6)

        for m in range(3):
            SubTN[m] = np.clip(Ganglia[m] + (1 / TaoSubTN) * (-Ganglia[m] + response[m] - Ganglia[m + 3] + np.dot(-weights_Gan[m, :], Ganglia[6:9])), 0, None)
            GPi[m] = np.clip((Ganglia[m + 3] + (1 / TaoGPi) * (-Ganglia[m + 3] + np.dot(weights_Gan[m, :], Ganglia[0:3]) - Ganglia[m + 6] - Ganglia[m + 9] - Ganglia[12] * (m == 0))), 0, None)
            GPe[m] = np.clip((Ganglia[m + 6] + (1 / TaoGPe) * (-Ganglia[m + 6] + Ganglia[m] - (Ganglia[13] + Ganglia[14]) * (m == 0))), 0, None)
            StR[m] = np.clip((Ganglia[m + 9] + (1 / TaoStR) * (-Ganglia[m + 9] + Ganglia[m] + stimSTR[m])), 0, None)
            StR[m + 3] = np.clip((Ganglia[m + 12] + (1 / TaoStR) * (-Ganglia[m + 12] + condSTR[m])), 0, None)

        return np.hstack([SubTN, GPi, GPe, StR])

    def cpg_propagation(self, spine, reticulospinal):
        dt = 1
        TauE = 1
        TauA = 3
        TauH = 15
        Tau1 = 1
        weights = np.array([[-1, 1], [1, -1], [1, -1]])
        EEl = np.clip(5 + np.sum(-weights * reticulospinal), 4, 5)
        EEr = np.clip(5 + np.sum(weights * reticulospinal), 4, 5)
        stimulus = np.clip(3 + np.sum(reticulospinal), 3, 7)
        data = np.zeros(15)

        stim = np.vstack([stimulus, spine[:14, 5].reshape(-1, 1)])
        for v in range(15):
            TauH = 15 / (1 + (0.2 * spine[v, 0]) ** 2)  # Efectos del estímulo en tau a través de 5-HT
            AHPgain = 9 + (0.09 * spine[v, 0]) ** 2
            # NEURONAS A
            spine[v, 0] = max(0, spine[v, 0] + (dt / TauA) * (-spine[v, 0] + stim[v]))
            data[v] = (stim[v] > 1.5) * (stim[v] < 2)
            # NEURONAS E
            spine[v, 1] = max(0, spine[v, 1] + data[v] + (dt / TauE) * (-spine[v, 1] + 8 * self.naka(spine[v, 0] + EEl * spine[v, 1] - 1.2 * spine[v, 2], 4.4 + AHPgain * spine[v, 3])))
            spine[v, 2] = max(0, spine[v, 2] + (dt / TauE) * (-spine[v, 2] + 8 * self.naka(spine[v, 0] + EEr * spine[v, 2] - 1.2 * spine[v, 1], 4.4 + AHPgain * spine[v, 4])))
            # NEURONAS H
            spine[v, 3] = max(0, spine[v, 3] + (dt / TauH) * (-spine[v, 3] + spine[v, 1]))
            spine[v, 4] = max(0, spine[v, 4] + (dt / TauH) * (-spine[v, 4] + spine[v, 2]))
            # NEURONAS DE RETARDO
            spine[v, 5] = max(0, spine[v, 5] + (dt / Tau1) * (-spine[v, 5] + spine[v, 0]))
            spine[v, 6] = stimulus * max(0, spine[v, 6] + (dt / 1) * (-spine[v, 6] + spine[v, 5] - (stimulus - 1)))

        return spine

    def publish_positions(self):
        
        if len(self.head_pose) != 6:
            self.get_logger().error(f"Invalid head_pose size: {len(self.head_pose)}. Expected 6 elements.")
            return

        for p in range(1):
            # Calculate the speed of the lamprey
            self.Lamprey = self.head_pose  #ARREGLO QUE LLEGA DESDE UNITY

            
            self.Estimulos = self.Estimulos = np.array(self.unity_coordinates).reshape(-1, 5) 
            #self.get_logger().info(f'ReceivedDD: {self.Estimulos}')

            speed = (self.spine[0, 1] + self.spine[0, 2]) * 0.05
            self.Lamprey[4] = self.Lamprey[4] + 0.05 * self.spine[0, 1] - 0.05 * self.spine[0, 2]
            
            # Simulate the retina and sensorimotor response
            retina, th, rho = self.retina_360(self.Estimulos[:, 1], self.Estimulos[:, 2], self.Lamprey, self.Estimulos[:, 0])
            self.modules, self.out = self.sensoriomotor(retina, self.modules, self.out, self.ganglios)
            self.ganglios = self.arbitramiento(np.sum(retina, axis=1), np.sum(self.modules[3:6, :], axis=1), self.ganglios, rho)
            self.spine = self.cpg_propagation(self.spine, self.out)

            # Update Lamprey position
            self.Lamprey[0] += int(speed * np.cos(np.radians(self.Lamprey[4])))
            self.Lamprey[1] += int(speed * np.sin(np.radians(self.Lamprey[4])))

            # Publish updated data
            msg = Float32MultiArray()
            msg.data = (3 * (self.spine[:, 1] - self.spine[:, 2])).flatten().astype(float).tolist()
            self.publisher_x.publish(msg)

            self.get_logger().info(f'Published: {msg.data}')

    def listener_callback_unity_coordinates(self, msg):
        self.unity_coordinates = msg.data
        #self.get_logger().info(f'Received unity coordinates: {self.unity_coordinates}')

    def listener_callback_get_pos(self, msg):
        self.head_pose = msg.data
        #self.get_logger().info(f'Received get_pos: {self.head_pose}')


def main(args=None):
    rclpy.init(args=args)
    position_publisher = PositionPublisher()
    rclpy.spin(position_publisher)
    position_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
