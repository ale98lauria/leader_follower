#!/usr/bin/env python3

import rclpy                         # Imports the client library for ROS2
from rclpy.node import Node          # Imports the Node class to create ROS2 nodes
import math                          # Imports the math module for mathematical functions
import threading                     # Imports the threading module to run threads in parallel
import time                          # Imports the time module to handle timing
import matplotlib.pyplot as plt      # Imports pyplot for plotting
import matplotlib.animation as animation  # Imports animation for real-time graph updates
from geometry_msgs.msg import Twist  # Imports the Twist message (for motion commands)
from nav_msgs.msg import Odometry    # Imports the Odometry message (for positions)
from scipy.spatial.transform import Rotation as R  # Imports the module for quaternion-to-Euler conversion

# Definition of the TurtleBotFollower class that extends Node
class TurtleBotFollower(Node):
    def __init__(self):
        # Initializes the node with the name 'turtlebot_follower'
        super().__init__('turtlebot_follower')
        
        # CONFIGURATION PARAMETERS
        self.leader_topic = '/tb1/odom'          # Topic from which to receive the leader's odometry
        self.follower_topic = '/tb2/odom'        # Topic from which to receive the follower's odometry
        self.follower_cmd_vel = '/tb2/cmd_vel'   # Topic on which to publish velocity commands for the follower
        self.offset_distance = 0.5               # Desired distance between leader and follower
        self.prediction_time = 0.8               # Prediction time to calculate the leader's future position
        self.stop_threshold = 0.02               # Threshold to consider the leader as stopped
        
        # PID CONTROLLER PARAMETERS
        self.kp_linear = 1.2                     # Proportional gain for linear velocity
        self.kd_linear = 0.4                     # Derivative gain for linear velocity
        self.kp_angular = 1.5                    # Proportional gain for angular velocity
        self.max_linear_speed = 0.2              # Maximum allowed linear speed
        self.max_angular_speed = 1.0             # Maximum allowed angular speed
        self.last_distance_error = 0.0           # Stores the previous iteration's distance error (for the derivative term)
        
        # ROS SUBSCRIPTIONS AND PUBLICATIONS
        self.leader_sub = self.create_subscription(
            Odometry, self.leader_topic, self.leader_callback, 10)
        self.follower_sub = self.create_subscription(
            Odometry, self.follower_topic, self.follower_callback, 10)
        self.cmd_vel_pub = self.create_publisher(Twist, self.follower_cmd_vel, 10)
        
        # STATE VARIABLES TO STORE POSITIONS AND ORIENTATIONS
        self.leader_x = self.leader_y = self.leader_yaw = self.leader_vx = 0.0
        self.follower_x = self.follower_y = self.follower_yaw = 0.0
        
        # LISTS TO STORE THE DATA FOR PLOTTING
        self.time_data = []             # Time elapsed since the start
        self.leader_x_data = []         # Leader's X coordinates
        self.leader_y_data = []         # Leader's Y coordinates
        self.follower_x_data = []       # Follower's X coordinates
        self.follower_y_data = []       # Follower's Y coordinates
        self.control_linear_data = []   # Linear speed values from the controller
        self.control_angular_data = []  # Angular speed values from the controller
        self.tracking_error_data = []   # Tracking error values
        self.start_time = time.time()   # Initial time for computing elapsed time
        
        # STARTS THE THREAD FOR REAL-TIME PLOTTING
        self.graph_thread = threading.Thread(target=self.run_graph, daemon=True)
        self.graph_thread.start()
    
    def leader_callback(self, msg):
        """Callback to update the leader's position and orientation."""
        self.leader_x = msg.pose.pose.position.x
        self.leader_y = msg.pose.pose.position.y
        self.leader_vx = msg.twist.twist.linear.x  # Leader's linear velocity
        
        # Converts the quaternion to Euler angles to get the orientation (yaw)
        quaternion = msg.pose.pose.orientation
        euler = R.from_quat([
            quaternion.x, quaternion.y, quaternion.z, quaternion.w
        ]).as_euler('xyz', degrees=False)
        self.leader_yaw = euler[2]  # The yaw angle is the third value
    
    def follower_callback(self, msg):
        """Callback to update the follower's position and compute the control command."""
        self.follower_x = msg.pose.pose.position.x
        self.follower_y = msg.pose.pose.position.y
        
        # Converts the quaternion to Euler angles to get the orientation
        quaternion = msg.pose.pose.orientation
        euler = R.from_quat([
            quaternion.x, quaternion.y, quaternion.z, quaternion.w
        ]).as_euler('xyz', degrees=False)
        self.follower_yaw = euler[2]
        
        # PREDICTION OF THE LEADER'S POSITION
        pred_x = self.leader_x + self.prediction_time * self.leader_vx * math.cos(self.leader_yaw)
        pred_y = self.leader_y + self.prediction_time * self.leader_vx * math.sin(self.leader_yaw)
        
        # CALCULATES THE TARGET POSITION FOR THE FOLLOWER
        target_x = pred_x - self.offset_distance * math.cos(self.leader_yaw)
        target_y = pred_y - self.offset_distance * math.sin(self.leader_yaw)
        
        # CALCULATES THE ERROR BETWEEN THE TARGET POSITION AND THE FOLLOWER'S CURRENT POSITION
        error_x = target_x - self.follower_x
        error_y = target_y - self.follower_y
        distance = math.sqrt(error_x**2 + error_y**2)
        distance_error = distance - self.offset_distance
        
        # CALCULATES AND STORES THE TRACKING ERROR
        tracking_error = abs(distance_error)
        self.tracking_error_data.append(tracking_error)
        
        # Checks if the leader is considered stopped
        leader_stopped = abs(self.leader_vx) < self.stop_threshold
        
        # PID CONTROL CALCULATION FOR LINEAR VELOCITY
        derivative = distance_error - self.last_distance_error  # Derivative term
        linear_speed = self.kp_linear * distance_error + self.kd_linear * derivative
        self.last_distance_error = distance_error
        
        # CALCULATES THE DESIRED ORIENTATION FOR THE FOLLOWER
        desired_yaw = math.atan2(error_y, error_x)
        yaw_error = desired_yaw - self.follower_yaw
        # Normalizes the angular error within the [-pi, pi] range
        yaw_error = math.atan2(math.sin(yaw_error), math.cos(yaw_error))
        
        # CREATES THE TWIST COMMAND MESSAGE
        twist_msg = Twist()
        if leader_stopped and distance < self.offset_distance + 0.1:
            twist_msg.linear.x = 0.0  # Stops the follower if the leader is stopped and the distance is almost correct
        else:
            twist_msg.linear.x = max(0.0, min(self.max_linear_speed, linear_speed))
        
        twist_msg.angular.z = max(-self.max_angular_speed, 
                                  min(self.max_angular_speed, self.kp_angular * yaw_error))
        
        # PUBLISHES THE VELOCITY COMMAND FOR THE FOLLOWER
        self.cmd_vel_pub.publish(twist_msg)
        
        # SAVES THE DATA FOR THE GRAPHS (time, positions, and control actions)
        current_time = time.time() - self.start_time
        self.time_data.append(current_time)
        self.leader_x_data.append(self.leader_x)
        self.leader_y_data.append(self.leader_y)
        self.follower_x_data.append(self.follower_x)
        self.follower_y_data.append(self.follower_y)
        self.control_linear_data.append(twist_msg.linear.x)
        self.control_angular_data.append(twist_msg.angular.z)
    
    def run_graph(self):
        """Function executed in a separate thread to display real-time graphs.
           It creates 4 separate figures for:
             1. XY trajectories (leader and follower)
             2. X and Y trajectories over time
             3. Control actions (linear and angular)
             4. Tracking error over time
        """
        plt.style.use('seaborn-darkgrid')
        
        # FIGURE 1: XY trajectories in two separate subplots
        fig_xy, (ax_xy_leader, ax_xy_follower) = plt.subplots(1, 2, figsize=(12, 6))
        def update_xy(frame):
            ax_xy_leader.clear()
            ax_xy_follower.clear()
            
            # Plot of the leader's trajectory
            ax_xy_leader.plot(self.leader_x_data, self.leader_y_data, label="Leader", color='blue')
            ax_xy_leader.set_title("Traiettoria XY Leader")
            ax_xy_leader.set_xlabel("X (m)")
            ax_xy_leader.set_ylabel("Y (m)")
            ax_xy_leader.legend()
            
            # Plot of the follower's trajectory
            ax_xy_follower.plot(self.follower_x_data, self.follower_y_data, label="Follower", color='green')
            ax_xy_follower.set_title("Traiettoria XY Follower")
            ax_xy_follower.set_xlabel("X (m)")
            ax_xy_follower.set_ylabel("Y (m)")
            ax_xy_follower.legend()
            
            # Saves the image every 10 frames
            if frame % 10 == 0:
                fig_xy.savefig("xy_trajectories.png")
        ani_xy = animation.FuncAnimation(fig_xy, update_xy, interval=500)
        
        # FIGURE 2: X and Y trajectories over time in two side-by-side subplots
        fig_time, (ax_time_x, ax_time_y) = plt.subplots(1, 2, figsize=(12, 6))
        def update_time(frame):
            ax_time_x.clear()
            ax_time_y.clear()
            
            # Plot of the X coordinate over time
            ax_time_x.plot(self.time_data, self.leader_x_data, label="Leader X", color='blue')
            ax_time_x.plot(self.time_data, self.follower_x_data, label="Follower X", color='green')
            ax_time_x.set_title("Traiettoria X nel Tempo")
            ax_time_x.set_xlabel("Tempo (s)")
            ax_time_x.set_ylabel("X (m)")
            ax_time_x.legend()
            
            # Plot of the Y coordinate over time
            ax_time_y.plot(self.time_data, self.leader_y_data, label="Leader Y", color='blue')
            ax_time_y.plot(self.time_data, self.follower_y_data, label="Follower Y", color='green')
            ax_time_y.set_title("Traiettoria Y nel Tempo")
            ax_time_y.set_xlabel("Tempo (s)")
            ax_time_y.set_ylabel("Y (m)")
            ax_time_y.legend()
            
            if frame % 10 == 0:
                fig_time.savefig("time_trajectories.png")
        ani_time = animation.FuncAnimation(fig_time, update_time, interval=500)
        
        # FIGURE 3: Control actions (linear and angular) in two vertical subplots
        fig_control, (ax_control_linear, ax_control_angular) = plt.subplots(2, 1, figsize=(8, 8))
        def update_control(frame):
            ax_control_linear.clear()
            ax_control_angular.clear()
            
            # Plot of the linear control action over time
            ax_control_linear.plot(self.time_data, self.control_linear_data, label="Controllo Lineare", color='purple')
            ax_control_linear.set_title("Azione di Controllo Lineare")
            ax_control_linear.set_xlabel("Tempo (s)")
            ax_control_linear.set_ylabel("Velocità Lineare (m/s)")
            ax_control_linear.legend()
            
            # Plot of the angular control action over time
            ax_control_angular.plot(self.time_data, self.control_angular_data, label="Controllo Angolare", color='orange')
            ax_control_angular.set_title("Azione di Controllo Angolare")
            ax_control_angular.set_xlabel("Tempo (s)")
            ax_control_angular.set_ylabel("Velocità Angolare (rad/s)")
            ax_control_angular.legend()
            
            if frame % 10 == 0:
                fig_control.savefig("control_actions.png")
        ani_control = animation.FuncAnimation(fig_control, update_control, interval=500)
        
        # FIGURE 4: Tracking error over time
        fig_error, ax_error = plt.subplots(figsize=(8, 6))
        def update_error(frame):
            ax_error.clear()
            # Plot of the tracking error (distance between target and actual position)
            ax_error.plot(self.time_data, self.tracking_error_data, label="Tracking Error", color='red')
            ax_error.set_title("Errore di Tracciamento nel Tempo")
            ax_error.set_xlabel("Tempo (s)")
            ax_error.set_ylabel("Errore (m)")
            ax_error.legend()
            if frame % 10 == 0:
                fig_error.savefig("tracking_error.png")
        ani_error = animation.FuncAnimation(fig_error, update_error, interval=500)
        
        # Displays all the created figures
        plt.show()

def main(args=None):
    # Initializes ROS2
    rclpy.init(args=args)
    # Creates an instance of the TurtleBotFollower node
    node = TurtleBotFollower()
    # Keeps the node running until it is shut down
    rclpy.spin(node)
    # Cleans up resources before ending
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
