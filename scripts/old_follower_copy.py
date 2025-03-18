#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import math
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
#from sensor_msgs.msg import LaserScan
from scipy.spatial.transform import Rotation as R

class TurtleBotFollower(Node):
    def __init__(self):
        super().__init__('turtlebot_follower')

        # Parametri
        self.leader_topic = '/tb1/odom'  # Odometria del leader
        self.follower_topic = '/tb2/odom'  # Odometria del follower
        self.follower_cmd_vel = '/tb2/cmd_vel'  # Comandi velocità del follower
        #self.lidar_topic= '/tb2/scan' # Sensore lidar per evitare ostacoli
        self.offset_distance = 0.5  # Offset di distanza dal leader
        self.prediction_time = 0.8 # Tempo di predizione del moto del leader
        self.stop_threshold = 0.02  # Soglia per considerare il leader fermo
        # Guadagni di controllo (PID) 
        self.kp_linear = 1.2  # Guadagno proporzionale per il controllo lineare
        self.kd_linear = 0.4  # Guadagno derivativo per la stabilizzazione
        self.kp_angular = 1.5  # Guadagno proporzionale per il controllo angolare

        self.max_linear_speed = 0.2  # Velocità massima lineare per evitare oscillazioni
        self.max_angular_speed = 1.0 # Velocità massima angolare
        self.last_distance_error = 0.0  # Per il controllo derivativo
        #self.safe_distance = 0.3  # Distanza minima di sicurezza dagli ostacoli

        # Subscriber per la posizione del leader e del follower
        self.leader_sub = self.create_subscription(Odometry, self.leader_topic, self.leader_callback, 10)
        self.follower_sub = self.create_subscription(Odometry, self.follower_topic, self.follower_callback, 10)
        #self.lidar_sub = self.create_subscription(LaserScan, self.lidar_topic, self.lidar_callback, 10)

        # Publisher per il movimento del follower
        self.cmd_vel_pub = self.create_publisher(Twist, self.follower_cmd_vel, 10)

        # Posizioni del leader e del follower
        self.leader_x = 0.0
        self.leader_y = 0.0
        self.leader_yaw = 0.0
        self.leader_vx = 0.0  # Velocità lineare del leader

        self.follower_x = 0.0
        self.follower_y = 0.0
        self.follower_yaw = 0.0
        #self.lidar_ranges = []  # Dati del Lidar per il rilevamento degli ostacoli

    def leader_callback(self, msg):
        """Callback per aggiornare la posizione e l'orientamento del leader."""
        self.leader_x = msg.pose.pose.position.x
        self.leader_y = msg.pose.pose.position.y
        
        # Estrazione della velocità del leader
        self.leader_vx = msg.twist.twist.linear.x

        # Estrarre l'angolo yaw del leader
        quaternion = msg.pose.pose.orientation
        quaternion_array = [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
        euler = R.from_quat(quaternion_array).as_euler('xyz', degrees=False)
        self.leader_yaw = euler[2]  # Yaw (rotazione attorno all'asse Z)

    def follower_callback(self, msg):
        """Callback per controllare il movimento del follower in base alla posizione del leader."""
        self.follower_x = msg.pose.pose.position.x
        self.follower_y = msg.pose.pose.position.y

        # Estrarre l'angolo yaw del follower
        quaternion = msg.pose.pose.orientation
        quaternion_array = [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
        euler = R.from_quat(quaternion_array).as_euler('xyz', degrees=False)
        self.follower_yaw = euler[2]  # Yaw

        # Predizione della posizione futura del leader**
        pred_x = self.leader_x + self.prediction_time * self.leader_vx * math.cos(self.leader_yaw)
        pred_y = self.leader_y + self.prediction_time * self.leader_vx * math.sin(self.leader_yaw)

        # Calcolare la posizione target per il follower (dietro al leader)
        
        #target_x = self.leader_x - self.offset_distance * math.cos(self.leader_yaw)
        #target_y = self.leader_y - self.offset_distance * math.sin(self.leader_yaw)
        target_x = pred_x - self.offset_distance * math.cos(self.leader_yaw)
        target_y = pred_y - self.offset_distance * math.sin(self.leader_yaw)

        # Calcolare l'errore di posizione
        error_x = target_x - self.follower_x
        error_y = target_y - self.follower_y
        distance = math.sqrt(error_x ** 2 + error_y ** 2)
        distance_error = distance - self.offset_distance

        # **Rilevamento STOP del leader**
        leader_stopped = abs(self.leader_vx) < self.stop_threshold

        # **Controllo PID sulla velocità lineare**
        derivative = distance_error - self.last_distance_error
        linear_speed = self.kp_linear * distance_error + self.kd_linear * derivative
        self.last_distance_error = distance_error

        # Calcolare l'angolo desiderato per il follower e dell'errore angolare
        desired_yaw = math.atan2(error_y, error_x)
        yaw_error = desired_yaw - self.follower_yaw
        yaw_error = math.atan2(math.sin(yaw_error), math.cos(yaw_error))

        # Normalizzare l'errore angolare nell'intervallo [-pi, pi]
        #yaw_error = math.atan2(math.sin(yaw_error), math.cos(yaw_error))

        # Creare il messaggio Twist
        twist_msg = Twist()
        # **Calcolo delle velocità** + **Logica di STOP SICURO**
        #twist_msg.linear.x = max(0.0, min(self.max_linear_speed, linear_speed))  # Limita velocità max
        if leader_stopped and distance < self.offset_distance + 0.1:
            twist_msg.linear.x = 0.0  # Blocca il movimento se il leader è fermo e siamo troppo vicini
        else:
            twist_msg.linear.x = max(0.0, min(self.max_linear_speed, linear_speed))  # Limita velocità max

        # **Regolazione angolare fluida**
        twist_msg.angular.z = max(-self.max_angular_speed, min(self.max_angular_speed, self.kp_angular * yaw_error))
        #twist_msg.angular.z = max(-self.max_angular_speed, min(self.max_angular_speed, self.kp_angular * yaw_error))

        # Se l'errore angolare è significativo, ruota prima di avanzare
        #if abs(yaw_error) > 0.1:
        #    twist_msg.angular.z = self.kp_angular * yaw_error
        #    twist_msg.linear.x = 0.0
        #else:
        #    # Se siamo abbastanza allineati, muoviti in avanti
        #    if distance > 0.05:
        #        #twist_msg.linear.x = self.kp_linear * distance
        #        twist_msg.linear.x = self.kp_linear * distance * (1 - abs(yaw_error))
        #        twist_msg.linear.x = min(self.max_linear_speed, twist_msg.linear.x)  # Limitare velocità massima
        #    else:
        #        twist_msg.linear.x = 0.0
        #    twist_msg.angular.z = self.kp_angular * yaw_error
        
        # Evitare collisioni con ostacoli usando il Lidar**
        #if self.lidar_ranges:
        #    min_distance = min(self.lidar_ranges)
        #    if min_distance < self.safe_distance:
        #        twist_msg.linear.x *= 0.5  # Rallenta in presenza di ostacoli vicini
        #        self.get_logger().warn("Ostacolo vicino! Riduzione velocità.")

        # Pubblica il comando di velocità
        self.cmd_vel_pub.publish(twist_msg)
    
    #def lidar_callback(self, msg):
    #    """Callback per ricevere i dati del Lidar e memorizzare le distanze."""
    #    self.lidar_ranges = [r for r in msg.ranges if not math.isnan(r)]

def main(args=None):
    rclpy.init(args=args)
    node = TurtleBotFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
