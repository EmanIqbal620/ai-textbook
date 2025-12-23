---
sidebar_position: 3
---

# Week 2: Python Integration and Node Development

This week focuses on Python integration with ROS 2 using the rclpy library. You'll learn to create publisher and subscriber nodes, implement service clients and servers, and manage node parameters.

## Learning Objectives

By the end of this week, you will be able to:

- Create publisher and subscriber nodes using rclpy
- Implement service clients and servers
- Manage parameters and node lifecycle
- Create more complex node interactions

## 2.1 Introduction to rclpy Library

rclpy is the Python client library for ROS 2. It provides the Python API for developing ROS 2 applications and handles the communication between nodes.

### Key Components of rclpy

- **Node**: The basic execution unit
- **Publisher**: Sends messages to topics
- **Subscriber**: Receives messages from topics
- **Client**: Calls services
- **Service**: Provides services
- **Timer**: Executes callbacks at regular intervals
- **Parameter**: Handles configuration values

### Basic Node Structure

```python
import rclpy
from rclpy.node import Node

class MyNode(Node):
    def __init__(self):
        # Initialize the node with a name
        super().__init__('my_node_name')

        # Node initialization code goes here
        self.get_logger().info('Node initialized')

def main(args=None):
    # Initialize ROS 2
    rclpy.init(args=args)

    # Create node instance
    node = MyNode()

    # Spin the node (keep it running)
    rclpy.spin(node)

    # Cleanup
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 2.2 Creating Publisher and Subscriber Nodes

### Advanced Publisher Example

Let's create a publisher that sends sensor data:

```python
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from sensor_msgs.msg import JointState
import random

class SensorPublisher(Node):
    def __init__(self):
        super().__init__('sensor_publisher')

        # Create publishers for different sensor types
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.temperature_pub = self.create_publisher(Float32, 'temperature', 10)

        # Create a timer to publish data at regular intervals
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.publish_sensor_data)

        # Initialize joint names for a humanoid robot
        self.joint_names = [
            'left_hip', 'left_knee', 'left_ankle',
            'right_hip', 'right_knee', 'right_ankle',
            'left_shoulder', 'left_elbow', 'left_wrist',
            'right_shoulder', 'right_elbow', 'right_wrist'
        ]

        self.get_logger().info('Sensor publisher node started')

    def publish_sensor_data(self):
        # Publish joint states
        joint_msg = JointState()
        joint_msg.name = self.joint_names
        joint_msg.position = [random.uniform(-1.5, 1.5) for _ in self.joint_names]
        joint_msg.velocity = [random.uniform(-0.5, 0.5) for _ in self.joint_names]
        joint_msg.effort = [random.uniform(0, 10) for _ in self.joint_names]

        self.joint_pub.publish(joint_msg)

        # Publish temperature
        temp_msg = Float32()
        temp_msg.data = random.uniform(20.0, 35.0)
        self.temperature_pub.publish(temp_msg)

        self.get_logger().info(f'Published joint states and temperature: {temp_msg.data:.2f}°C')

def main(args=None):
    rclpy.init(args=args)
    node = SensorPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Advanced Subscriber Example

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32

class SensorSubscriber(Node):
    def __init__(self):
        super().__init__('sensor_subscriber')

        # Create subscribers
        self.joint_sub = self.create_subscription(
            JointState,
            'joint_states',
            self.joint_callback,
            10
        )

        self.temperature_sub = self.create_subscription(
            Float32,
            'temperature',
            self.temperature_callback,
            10
        )

        self.get_logger().info('Sensor subscriber node started')

    def joint_callback(self, msg):
        self.get_logger().info(f'Received {len(msg.name)} joints')
        # Process joint data
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.get_logger().info(f'{name}: pos={msg.position[i]:.2f}, vel={msg.velocity[i]:.2f}')

    def temperature_callback(self, msg):
        self.get_logger().info(f'Current temperature: {msg.data:.2f}°C')

def main(args=None):
    rclpy.init(args=args)
    node = SensorSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 2.3 Service Clients and Servers

### Service Server Example

First, create a service definition file `GetRobotStatus.srv` in a `srv` directory:

```
# Request
---
# Response
string status
float32 battery_level
bool is_operational
```

Then implement the service server:

```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import Trigger  # Using built-in service for example
import random

class RobotStatusService(Node):
    def __init__(self):
        super().__init__('robot_status_service')

        # Create a service
        self.srv = self.create_service(
            Trigger,  # Using Trigger as example - you would use your custom service
            'get_robot_status',
            self.status_callback
        )

        self.get_logger().info('Robot status service started')

    def status_callback(self, request, response):
        # Simulate getting robot status
        response.success = True
        response.message = f'Robot operational at {random.uniform(80, 100):.1f}%'

        self.get_logger().info(f'Service called, returning: {response.message}')
        return response

def main(args=None):
    rclpy.init(args=args)
    node = RobotStatusService()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Service Client Example

```python
import rclpy
from rclpy.node import Node
from example_interfaces.srv import Trigger
import sys

class StatusClient(Node):
    def __init__(self):
        super().__init__('status_client')
        self.cli = self.create_client(Trigger, 'get_robot_status')

        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

        self.req = Trigger.Request()

    def send_request(self):
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

def main(args=None):
    rclpy.init(args=args)
    client = StatusClient()
    response = client.send_request()

    if response is not None:
        client.get_logger().info(f'Response: {response.success}, {response.message}')
    else:
        client.get_logger().info('Service call failed')

    client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 2.4 Parameter Management and Node Lifecycle

### Parameter Declaration and Usage

```python
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

class ParameterNode(Node):
    def __init__(self):
        super().__init__('parameter_node')

        # Declare parameters with default values
        self.declare_parameter('robot_name', 'humanoid_robot')
        self.declare_parameter('max_velocity', 1.0)
        self.declare_parameter('safety_distance', 0.5)

        # Get parameter values
        self.robot_name = self.get_parameter('robot_name').value
        self.max_velocity = self.get_parameter('max_velocity').value
        self.safety_distance = self.get_parameter('safety_distance').value

        self.get_logger().info(f'Robot name: {self.robot_name}')
        self.get_logger().info(f'Max velocity: {self.max_velocity}')
        self.get_logger().info(f'Safety distance: {self.safety_distance}')

        # Set up parameter callback for dynamic changes
        self.add_on_set_parameters_callback(self.parameter_callback)

    def parameter_callback(self, params):
        for param in params:
            if param.name == 'max_velocity' and param.value > 5.0:
                return SetParametersResult(successful=False, reason='Max velocity too high')
        return SetParametersResult(successful=True)

from rclpy.parameter_service import SetParametersResult

def main(args=None):
    rclpy.init(args=args)
    node = ParameterNode()

    # You can change parameters at runtime
    # ros2 param set /parameter_node max_velocity 2.0

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

### Node Lifecycle Management

```python
import rclpy
from rclpy.node import Node
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
import time

class LifecycleRobotNode(LifecycleNode):
    def __init__(self):
        super().__init__('lifecycle_robot_node')
        self.get_logger().info('Lifecycle node created, current state: unconfigured')

    def on_configure(self, state):
        self.get_logger().info('Configuring lifecycle node')
        # Initialize resources here
        self.publisher = self.create_publisher(String, 'robot_status', 10)
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        self.get_logger().info('Activating lifecycle node')
        # Start operations here
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        self.get_logger().info('Deactivating lifecycle node')
        # Pause operations here
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state):
        self.get_logger().info('Cleaning up lifecycle node')
        # Clean up resources here
        self.publisher = None
        return TransitionCallbackReturn.SUCCESS

def main(args=None):
    rclpy.init(args=args)
    node = LifecycleRobotNode()

    # Use lifecycle node manager
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 2.5 Practical Exercise: Building a Sensor Data Publisher

Create a comprehensive node that publishes multiple types of sensor data for a humanoid robot:

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu, LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray
import random
import math

class HumanoidSensorPublisher(Node):
    def __init__(self):
        super().__init__('humanoid_sensor_publisher')

        # Create publishers for different sensor types
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)
        self.imu_pub = self.create_publisher(Imu, 'imu/data', 10)
        self.laser_pub = self.create_publisher(LaserScan, 'scan', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # Timer for publishing sensor data
        timer_period = 0.05  # 20 Hz
        self.timer = self.create_timer(timer_period, self.publish_sensor_data)

        # Initialize humanoid joint names
        self.joint_names = [
            'left_hip_joint', 'left_knee_joint', 'left_ankle_joint',
            'right_hip_joint', 'right_knee_joint', 'right_ankle_joint',
            'left_shoulder_joint', 'left_elbow_joint', 'left_wrist_joint',
            'right_shoulder_joint', 'right_elbow_joint', 'right_wrist_joint',
            'head_pan_joint', 'head_tilt_joint'
        ]

        self.get_logger().info('Humanoid sensor publisher started')

    def publish_sensor_data(self):
        # Publish joint states
        joint_msg = JointState()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.name = self.joint_names
        joint_msg.position = [random.uniform(-1.5, 1.5) for _ in self.joint_names]
        joint_msg.velocity = [random.uniform(-0.5, 0.5) for _ in self.joint_names]
        joint_msg.effort = [random.uniform(0, 10) for _ in self.joint_names]
        self.joint_pub.publish(joint_msg)

        # Publish IMU data
        imu_msg = Imu()
        imu_msg.header.stamp = self.get_clock().now().to_msg()
        imu_msg.header.frame_id = 'imu_link'
        # Simulate orientation (simplified)
        imu_msg.orientation.x = random.uniform(-0.1, 0.1)
        imu_msg.orientation.y = random.uniform(-0.1, 0.1)
        imu_msg.orientation.z = random.uniform(-0.01, 0.01)
        imu_msg.orientation.w = 1.0  # Normalize
        self.imu_pub.publish(imu_msg)

        # Publish laser scan
        scan_msg = LaserScan()
        scan_msg.header.stamp = self.get_clock().now().to_msg()
        scan_msg.header.frame_id = 'laser_frame'
        scan_msg.angle_min = -math.pi / 2
        scan_msg.angle_max = math.pi / 2
        scan_msg.angle_increment = math.pi / 180  # 1 degree
        scan_msg.time_increment = 0.0
        scan_msg.scan_time = 0.1
        scan_msg.range_min = 0.1
        scan_msg.range_max = 10.0
        num_ranges = int((scan_msg.angle_max - scan_msg.angle_min) / scan_msg.angle_increment)
        scan_msg.ranges = [random.uniform(0.5, 5.0) for _ in range(num_ranges)]
        self.laser_pub.publish(scan_msg)

        self.get_logger().info('Published sensor data')

def main(args=None):
    rclpy.init(args=args)
    node = HumanoidSensorPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## 2.6 Mind Map: Connecting ROS 2 Concepts, Python Code, and URDF

```mermaid
mindmap
  root((ROS 2 Concepts))
    Nodes
      Publisher
        rclpy.create_publisher()
        Publish messages
        Topic-based communication
      Subscriber
        rclpy.create_subscription()
        Receive messages
        Callback functions
      Service Server
        rclpy.create_service()
        Request/Response pattern
      Service Client
        rclpy.create_client()
        Call services
    Messages
      Standard Messages
        std_msgs
        geometry_msgs
        sensor_msgs
      Custom Messages
        Define .msg files
        Generate message types
    Parameters
      Declare Parameters
        declare_parameter()
        Get/Set parameters
      Dynamic Reconfiguration
        Parameter callbacks
        Runtime updates
    Python Integration
      rclpy library
        Node class
        Publishers/Subscribers
        Services/Actions
      ROS 2 Client Library
        C++ backend
        Python bindings
    URDF Connection
      Robot Description
        Joint definitions
        Link properties
      Joint State Publisher
        Publish joint positions
        Synchronize with simulation
```

## Summary

This week focused on Python integration with ROS 2 using rclpy. You learned to create complex publisher and subscriber nodes, implement services, manage parameters, and handle node lifecycles. These skills are essential for developing humanoid robot applications.

## Next Week Preview

Next week, we'll explore URDF (Unified Robot Description Format) and learn how to create and manipulate robot models for humanoid robots.