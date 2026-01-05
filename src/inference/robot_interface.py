# src/inference/robot_interface.py
"""
Robot Communication Interface for Vision System

Features:
1. Real-time robot state monitoring
2. Command handling and execution
3. Safety constraints enforcement
4. Network communication protocols
5. Emergency stop handling
"""

import socket
import threading
import time
import json
import struct
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import queue
import pickle
import zlib
import asyncio
from concurrent.futures import ThreadPoolExecutor
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore', category=UserWarning)

@dataclass
class RobotConfig:
    """Robot configuration parameters."""
    # Network settings
    robot_ip: str = "192.168.1.100"
    robot_port: int = 5000
    vision_ip: str = "0.0.0.0"
    vision_port: int = 5001
    
    # Communication protocols
    protocol: str = "tcp"  # tcp, udp, ros, zmq
    heartbeat_interval: float = 1.0  # seconds
    timeout: float = 5.0  # seconds
    reconnect_attempts: int = 3
    
    # Robot parameters
    max_linear_velocity: float = 1.0  # m/s
    max_angular_velocity: float = 1.0  # rad/s
    safety_distance: float = 0.5  # meters
    emergency_stop_distance: float = 0.2  # meters
    
    # Vision parameters
    camera_height: float = 1.5  # meters
    camera_tilt: float = 0.0  # radians
    camera_fov: Tuple[float, float] = (1.047, 0.785)  # (horizontal, vertical) radians
    
    # Command limits
    max_commands_per_second: int = 10
    command_queue_size: int = 100

class RobotState(Enum):
    """Robot operational states."""
    INITIALIZING = "initializing"
    READY = "ready"
    MOVING = "moving"
    STOPPED = "stopped"
    EMERGENCY_STOP = "emergency_stop"
    ERROR = "error"
    MAINTENANCE = "maintenance"

@dataclass
class RobotStatus:
    """Robot status information."""
    state: RobotState = RobotState.INITIALIZING
    battery_level: float = 100.0  # percentage
    temperature: float = 25.0  # degrees Celsius
    uptime: float = 0.0  # seconds
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # x, y, theta
    velocity: Tuple[float, float] = (0.0, 0.0)  # linear, angular
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []

@dataclass
class DetectionCommand:
    """Command based on detection results."""
    detection_id: int
    class_id: int
    confidence: float
    position: Tuple[float, float, float]  # x, y, z in robot coordinates
    action: str  # approach, avoid, track, ignore
    priority: int = 1  # 1=lowest, 10=highest
    timestamp: float = 0.0

@dataclass
class RobotCommand:
    """Robot movement command."""
    command_type: str  # move, stop, rotate, custom
    linear_velocity: float = 0.0
    angular_velocity: float = 0.0
    duration: float = 0.0  # seconds, 0=until next command
    target_position: Optional[Tuple[float, float, float]] = None
    safety_check: bool = True
    emergency_override: bool = False
    timestamp: float = 0.0

class RobotCommunication:
    """
    Robot communication interface for vision system.
    
    Features:
    1. Bidirectional communication with robot
    2. Real-time state monitoring
    3. Command execution with safety checks
    4. Emergency handling
    5. Multiple protocol support
    """
    
    def __init__(self, config: Optional[RobotConfig] = None):
        """
        Initialize robot communication interface.
        
        Args:
            config: Robot configuration
        """
        self.config = config or RobotConfig()
        self.status = RobotStatus()
        
        # Communication
        self.socket = None
        self.connected = False
        self.connection_lock = threading.Lock()
        
        # Command handling
        self.command_queue = queue.Queue(maxsize=self.config.command_queue_size)
        self.command_history = []
        self.last_command_time = 0.0
        
        # Thread management
        self.communication_thread = None
        self.command_thread = None
        self.heartbeat_thread = None
        self.running = False
        
        # Safety monitoring
        self.safety_monitor = SafetyMonitor(self.config)
        self.emergency_stop = False
        
        # Performance monitoring
        self.message_count = 0
        self.last_update_time = time.time()
        
        # Initialize communication
        self._init_communication()
        
        logger.info("RobotCommunication initialized")
        logger.info(f"Robot IP: {self.config.robot_ip}:{self.config.robot_port}")
        logger.info(f"Protocol: {self.config.protocol}")
    
    def _init_communication(self):
        """Initialize communication based on protocol."""
        protocol = self.config.protocol.lower()
        
        if protocol == "tcp":
            self._init_tcp_communication()
        elif protocol == "udp":
            self._init_udp_communication()
        elif protocol == "ros":
            self._init_ros_communication()
        elif protocol == "zmq":
            self._init_zmq_communication()
        else:
            raise ValueError(f"Unsupported protocol: {protocol}")
    
    def _init_tcp_communication(self):
        """Initialize TCP communication."""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.settimeout(self.config.timeout)
    
    def _init_udp_communication(self):
        """Initialize UDP communication."""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.settimeout(self.config.timeout)
    
    def _init_ros_communication(self):
        """Initialize ROS communication."""
        try:
            import rospy
            from std_msgs.msg import String
            from geometry_msgs.msg import Twist
            
            self.ros_initialized = True
            self.ros_node = None
            self.command_publisher = None
            self.status_subscriber = None
            
            logger.info("ROS communication initialized")
        except ImportError:
            logger.warning("ROS not available, falling back to TCP")
            self.config.protocol = "tcp"
            self._init_tcp_communication()
    
    def _init_zmq_communication(self):
        """Initialize ZeroMQ communication."""
        try:
            import zmq
            
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.PAIR)
            self.socket.setsockopt(zmq.LINGER, 0)
            
            logger.info("ZeroMQ communication initialized")
        except ImportError:
            logger.warning("ZeroMQ not available, falling back to TCP")
            self.config.protocol = "tcp"
            self._init_tcp_communication()
    
    def connect(self) -> bool:
        """
        Connect to robot.
        
        Returns:
            True if connection successful
        """
        with self.connection_lock:
            if self.connected:
                logger.warning("Already connected to robot")
                return True
            
            logger.info(f"Connecting to robot at {self.config.robot_ip}:{self.config.robot_port}")
            
            try:
                if self.config.protocol in ["tcp", "udp"]:
                    self.socket.connect((self.config.robot_ip, self.config.robot_port))
                    self.connected = True
                    
                elif self.config.protocol == "ros":
                    import rospy
                    
                    if not self.ros_initialized:
                        rospy.init_node('vision_robot_interface', anonymous=True)
                        self.ros_node = rospy
                    
                    # Create publishers and subscribers
                    self.command_publisher = rospy.Publisher(
                        '/robot_command',
                        String,
                        queue_size=10
                    )
                    
                    self.status_subscriber = rospy.Subscriber(
                        '/robot_status',
                        String,
                        self._ros_status_callback
                    )
                    
                    self.connected = True
                    
                elif self.config.protocol == "zmq":
                    self.socket.connect(f"tcp://{self.config.robot_ip}:{self.config.robot_port}")
                    self.connected = True
                
                logger.info("Connected to robot successfully")
                
                # Start communication threads
                self._start_threads()
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to connect to robot: {e}")
                self.connected = False
                return False
    
    def disconnect(self):
        """Disconnect from robot."""
        with self.connection_lock:
            if not self.connected:
                return
            
            logger.info("Disconnecting from robot")
            
            # Stop threads
            self._stop_threads()
            
            # Close socket
            if self.socket:
                try:
                    self.socket.close()
                except:
                    pass
            
            self.connected = False
            logger.info("Disconnected from robot")
    
    def _start_threads(self):
        """Start communication threads."""
        if self.running:
            return
        
        self.running = True
        
        # Start communication thread
        self.communication_thread = threading.Thread(
            target=self._communication_loop,
            daemon=True
        )
        self.communication_thread.start()
        
        # Start command processing thread
        self.command_thread = threading.Thread(
            target=self._command_processing_loop,
            daemon=True
        )
        self.command_thread.start()
        
        # Start heartbeat thread
        self.heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True
        )
        self.heartbeat_thread.start()
        
        logger.info("Communication threads started")
    
    def _stop_threads(self):
        """Stop communication threads."""
        self.running = False
        
        # Wait for threads to finish
        threads = [
            self.communication_thread,
            self.command_thread,
            self.heartbeat_thread
        ]
        
        for thread in threads:
            if thread and thread.is_alive():
                thread.join(timeout=2.0)
        
        logger.info("Communication threads stopped")
    
    def _communication_loop(self):
        """Main communication loop."""
        logger.info("Communication loop started")
        
        while self.running and self.connected:
            try:
                # Receive data from robot
                if self.config.protocol in ["tcp", "udp"]:
                    data = self._receive_socket_data()
                elif self.config.protocol == "ros":
                    # ROS uses callbacks, nothing to do here
                    time.sleep(0.01)
                    continue
                elif self.config.protocol == "zmq":
                    data = self._receive_zmq_data()
                
                if data:
                    self._process_incoming_data(data)
                
                # Small sleep to prevent CPU spinning
                time.sleep(0.001)
                
            except socket.timeout:
                continue
            except Exception as e:
                logger.error(f"Error in communication loop: {e}")
                time.sleep(0.1)
        
        logger.info("Communication loop stopped")
    
    def _receive_socket_data(self) -> Optional[bytes]:
        """Receive data from socket."""
        try:
            # First, receive message length (4 bytes)
            length_data = self.socket.recv(4)
            if not length_data:
                return None
            
            message_length = struct.unpack('>I', length_data)[0]
            
            # Receive message data
            chunks = []
            bytes_received = 0
            
            while bytes_received < message_length:
                chunk = self.socket.recv(min(message_length - bytes_received, 4096))
                if not chunk:
                    raise ConnectionError("Connection closed")
                
                chunks.append(chunk)
                bytes_received += len(chunk)
            
            return b''.join(chunks)
            
        except socket.timeout:
            return None
        except Exception as e:
            logger.error(f"Error receiving socket data: {e}")
            return None
    
    def _receive_zmq_data(self) -> Optional[bytes]:
        """Receive data from ZeroMQ socket."""
        try:
            message = self.socket.recv(zmq.NOBLOCK)
            return message
        except zmq.Again:
            return None
        except Exception as e:
            logger.error(f"Error receiving ZMQ data: {e}")
            return None
    
    def _ros_status_callback(self, msg):
        """ROS status callback."""
        try:
            data = json.loads(msg.data)
            self._update_status(data)
        except Exception as e:
            logger.error(f"Error in ROS status callback: {e}")
    
    def _process_incoming_data(self, data: bytes):
        """Process incoming data from robot."""
        try:
            # Decompress if needed
            if data.startswith(b'COMPRESSED'):
                data = zlib.decompress(data[10:])
            
            # Decode message
            message = json.loads(data.decode('utf-8'))
            
            # Update robot status
            if 'status' in message:
                self._update_status(message['status'])
            
            # Handle commands
            if 'command_response' in message:
                self._handle_command_response(message['command_response'])
            
            # Handle emergency stop
            if 'emergency_stop' in message and message['emergency_stop']:
                self._handle_emergency_stop()
            
            self.message_count += 1
            
        except Exception as e:
            logger.error(f"Error processing incoming data: {e}")
    
    def _update_status(self, status_data: Dict[str, Any]):
        """Update robot status from received data."""
        try:
            # Update basic status
            if 'state' in status_data:
                self.status.state = RobotState(status_data['state'])
            
            if 'battery_level' in status_data:
                self.status.battery_level = float(status_data['battery_level'])
            
            if 'temperature' in status_data:
                self.status.temperature = float(status_data['temperature'])
            
            if 'uptime' in status_data:
                self.status.uptime = float(status_data['uptime'])
            
            if 'position' in status_data:
                pos = status_data['position']
                self.status.position = (float(pos[0]), float(pos[1]), float(pos[2]))
            
            if 'velocity' in status_data:
                vel = status_data['velocity']
                self.status.velocity = (float(vel[0]), float(vel[1]))
            
            if 'errors' in status_data:
                self.status.errors = list(status_data['errors'])
            
            # Update safety monitor
            self.safety_monitor.update_robot_status(self.status)
            
            # Log status update
            if time.time() - self.last_update_time > 5.0:  # Log every 5 seconds
                logger.info(f"Robot Status: {self.status.state.value}, "
                          f"Battery: {self.status.battery_level:.1f}%, "
                          f"Position: {self.status.position}")
                self.last_update_time = time.time()
            
        except Exception as e:
            logger.error(f"Error updating robot status: {e}")
    
    def _handle_command_response(self, response: Dict[str, Any]):
        """Handle command response from robot."""
        command_id = response.get('command_id')
        success = response.get('success', False)
        message = response.get('message', '')
        
        if command_id:
            logger.info(f"Command {command_id} response: {success} - {message}")
            
            # Update command history
            self.command_history.append({
                'command_id': command_id,
                'success': success,
                'message': message,
                'timestamp': time.time()
            })
            
            # Keep only recent history
            if len(self.command_history) > 100:
                self.command_history.pop(0)
    
    def _handle_emergency_stop(self):
        """Handle emergency stop from robot."""
        logger.warning("Emergency stop received from robot!")
        self.emergency_stop = True
        self.status.state = RobotState.EMERGENCY_STOP
        
        # Clear command queue
        while not self.command_queue.empty():
            try:
                self.command_queue.get_nowait()
            except queue.Empty:
                break
    
    def _command_processing_loop(self):
        """Command processing loop."""
        logger.info("Command processing loop started")
        
        while self.running and self.connected:
            try:
                # Get command from queue
                command = self.command_queue.get(timeout=0.1)
                
                # Check rate limiting
                current_time = time.time()
                time_since_last_command = current_time - self.last_command_time
                
                if time_since_last_command < 1.0 / self.config.max_commands_per_second:
                    # Rate limit exceeded, wait
                    time.sleep(1.0 / self.config.max_commands_per_second - time_since_last_command)
                
                # Check safety constraints
                if not self._check_safety_constraints(command):
                    logger.warning(f"Command blocked by safety constraints: {command}")
                    continue
                
                # Send command to robot
                self._send_command(command)
                
                # Update last command time
                self.last_command_time = time.time()
                
                # Mark task as done
                self.command_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in command processing loop: {e}")
                time.sleep(0.1)
        
        logger.info("Command processing loop stopped")
    
    def _check_safety_constraints(self, command: RobotCommand) -> bool:
        """
        Check if command satisfies safety constraints.
        
        Args:
            command: Robot command to check
            
        Returns:
            True if command is safe
        """
        # Check emergency stop
        if self.emergency_stop and not command.emergency_override:
            logger.warning("Command blocked: Emergency stop active")
            return False
        
        # Check velocity limits
        if abs(command.linear_velocity) > self.config.max_linear_velocity:
            logger.warning(f"Linear velocity {command.linear_velocity} exceeds limit "
                         f"{self.config.max_linear_velocity}")
            return False
        
        if abs(command.angular_velocity) > self.config.max_angular_velocity:
            logger.warning(f"Angular velocity {command.angular_velocity} exceeds limit "
                         f"{self.config.max_angular_velocity}")
            return False
        
        # Check safety monitor
        if command.safety_check:
            safe = self.safety_monitor.check_command_safety(command)
            if not safe:
                logger.warning("Command blocked by safety monitor")
                return False
        
        return True
    
    def _send_command(self, command: RobotCommand):
        """Send command to robot."""
        try:
            # Prepare command data
            command_data = {
                'command': asdict(command),
                'timestamp': time.time(),
                'command_id': len(self.command_history) + 1
            }
            
            # Serialize and send
            message = json.dumps(command_data).encode('utf-8')
            
            if self.config.protocol in ["tcp", "udp"]:
                # Add message length prefix
                message_with_length = struct.pack('>I', len(message)) + message
                self.socket.sendall(message_with_length)
                
            elif self.config.protocol == "ros":
                import rospy
                from std_msgs.msg import String
                
                msg = String()
                msg.data = json.dumps(command_data)
                self.command_publisher.publish(msg)
                
            elif self.config.protocol == "zmq":
                self.socket.send(message)
            
            logger.debug(f"Command sent: {command.command_type}")
            
        except Exception as e:
            logger.error(f"Error sending command: {e}")
    
    def _heartbeat_loop(self):
        """Heartbeat loop to maintain connection."""
        logger.info("Heartbeat loop started")
        
        while self.running and self.connected:
            try:
                # Send heartbeat
                self._send_heartbeat()
                
                # Wait for interval
                time.sleep(self.config.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                time.sleep(1.0)
        
        logger.info("Heartbeat loop stopped")
    
    def _send_heartbeat(self):
        """Send heartbeat to robot."""
        heartbeat_data = {
            'heartbeat': True,
            'timestamp': time.time(),
            'vision_status': 'active'
        }
        
        try:
            message = json.dumps(heartbeat_data).encode('utf-8')
            
            if self.config.protocol in ["tcp", "udp"]:
                message_with_length = struct.pack('>I', len(message)) + message
                self.socket.sendall(message_with_length)
            elif self.config.protocol == "zmq":
                self.socket.send(message)
            
        except Exception as e:
            logger.error(f"Error sending heartbeat: {e}")
            self.connected = False
    
    def send_detection_command(self, detection: DetectionCommand) -> bool:
        """
        Send command based on detection.
        
        Args:
            detection: Detection command
            
        Returns:
            True if command queued successfully
        """
        # Create robot command from detection
        robot_command = self._detection_to_command(detection)
        
        if not robot_command:
            return False
        
        # Add to command queue
        try:
            self.command_queue.put_nowait(robot_command)
            return True
        except queue.Full:
            logger.warning("Command queue full, dropping detection command")
            return False
    
    def _detection_to_command(self, detection: DetectionCommand) -> Optional[RobotCommand]:
        """Convert detection to robot command."""
        if detection.action == "approach":
            # Approach the detected object
            command = RobotCommand(
                command_type="move",
                linear_velocity=self.config.max_linear_velocity * 0.5,  # 50% speed
                target_position=detection.position,
                safety_check=True
            )
            
        elif detection.action == "avoid":
            # Avoid the detected object
            # Simple avoidance: move away from object
            command = RobotCommand(
                command_type="rotate",
                angular_velocity=self.config.max_angular_velocity * 0.3,
                duration=1.0,
                safety_check=True
            )
            
        elif detection.action == "track":
            # Track the detected object (follow)
            command = RobotCommand(
                command_type="move",
                linear_velocity=self.config.max_linear_velocity * 0.3,
                safety_check=True
            )
            
        elif detection.action == "ignore":
            # No action needed
            return None
            
        else:
            logger.warning(f"Unknown detection action: {detection.action}")
            return None
        
        # Set timestamp
        command.timestamp = time.time()
        
        return command
    
    def emergency_stop_robot(self):
        """Send emergency stop command to robot."""
        logger.warning("Sending emergency stop command to robot")
        
        emergency_command = RobotCommand(
            command_type="stop",
            emergency_override=True,
            safety_check=False
        )
        
        # Clear queue and send emergency command
        while not self.command_queue.empty():
            try:
                self.command_queue.get_nowait()
            except queue.Empty:
                break
        
        try:
            self.command_queue.put_nowait(emergency_command)
            self.emergency_stop = True
            self.status.state = RobotState.EMERGENCY_STOP
        except queue.Full:
            logger.error("Failed to queue emergency stop command")
    
    def reset_emergency_stop(self):
        """Reset emergency stop state."""
        if self.emergency_stop:
            logger.info("Resetting emergency stop")
            self.emergency_stop = False
            
            if self.status.state == RobotState.EMERGENCY_STOP:
                self.status.state = RobotState.READY
    
    def get_status(self) -> RobotStatus:
        """Get current robot status."""
        return self.status
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get communication performance statistics."""
        stats = {
            'connected': self.connected,
            'message_count': self.message_count,
            'command_queue_size': self.command_queue.qsize(),
            'command_history_size': len(self.command_history),
            'emergency_stop': self.emergency_stop,
            'uptime': time.time() - self.last_update_time,
        }
        
        # Add safety monitor stats
        stats.update(self.safety_monitor.get_stats())
        
        return stats

class SafetyMonitor:
    """Safety monitor for robot commands."""
    
    def __init__(self, config: RobotConfig):
        self.config = config
        self.robot_status = None
        
        # Safety parameters
        self.min_safe_distance = config.safety_distance
        self.emergency_distance = config.emergency_stop_distance
        
        # Obstacle tracking
        self.obstacles = []  # List of (x, y, radius) in robot coordinates
        self.obstacle_history = deque(maxlen=100)
        
        # Safety violations
        self.violations = deque(maxlen=50)
        
        logger.info("SafetyMonitor initialized")
    
    def update_robot_status(self, status: RobotStatus):
        """Update robot status for safety monitoring."""
        self.robot_status = status
    
    def update_obstacles(self, detections: List[Dict[str, Any]]):
        """Update obstacle information from detections."""
        self.obstacles.clear()
        
        for detection in detections:
            # Convert detection to obstacle
            if 'position' in detection:
                pos = detection['position']
                
                # Estimate obstacle radius based on class
                radius = self._estimate_obstacle_radius(detection.get('class_id', 0))
                
                self.obstacles.append((pos[0], pos[1], radius))
        
        # Add to history
        self.obstacle_history.append({
            'timestamp': time.time(),
            'obstacles': self.obstacles.copy()
        })
    
    def _estimate_obstacle_radius(self, class_id: int) -> float:
        """Estimate obstacle radius based on class."""
        # Default sizes for common objects (in meters)
        object_sizes = {
            0: 0.3,   # person
            1: 0.5,   # bicycle
            2: 1.0,   # car
            3: 0.8,   # motorcycle
            5: 2.0,   # bus
            7: 1.5,   # truck
            56: 0.5,  # chair
            57: 1.0,  # couch
            62: 0.3,  # tv
        }
        
        return object_sizes.get(class_id, 0.5)  # Default 0.5m
    
    def check_command_safety(self, command: RobotCommand) -> bool:
        """
        Check if command is safe to execute.
        
        Args:
            command: Robot command to check
            
        Returns:
            True if command is safe
        """
        if not self.robot_status:
            logger.warning("Cannot check safety: Robot status not available")
            return True  # Allow if no status
        
        # Check for obstacles in path
        if command.command_type in ["move", "rotate"]:
            # Predict robot trajectory
            trajectory = self._predict_trajectory(command)
            
            # Check for collisions
            collision, distance = self._check_collision(trajectory)
            
            if collision:
                if distance < self.emergency_distance:
                    logger.error(f"Emergency stop: Collision imminent at {distance:.2f}m")
                    self._record_violation("collision_imminent", distance)
                    return False
                elif distance < self.min_safe_distance:
                    logger.warning(f"Safety violation: Too close to obstacle at {distance:.2f}m")
                    self._record_violation("too_close", distance)
                    return False
        
        return True
    
    def _predict_trajectory(self, command: RobotCommand) -> List[Tuple[float, float]]:
        """Predict robot trajectory based on command."""
        trajectory = []
        
        # Current position
        x, y, theta = self.robot_status.position
        
        # Time steps
        dt = 0.1  # 100ms steps
        total_time = command.duration if command.duration > 0 else 2.0  # Default 2s prediction
        
        for t in np.arange(0, total_time, dt):
            # Update position based on velocity
            if command.command_type == "move":
                x += command.linear_velocity * np.cos(theta) * dt
                y += command.linear_velocity * np.sin(theta) * dt
            elif command.command_type == "rotate":
                theta += command.angular_velocity * dt
            
            trajectory.append((x, y))
        
        return trajectory
    
    def _check_collision(self, trajectory: List[Tuple[float, float]]) -> Tuple[bool, float]:
        """Check for collisions along trajectory."""
        min_distance = float('inf')
        
        for point in trajectory:
            for obstacle in self.obstacles:
                obs_x, obs_y, obs_radius = obstacle
                
                # Calculate distance to obstacle
                distance = np.sqrt((point[0] - obs_x)**2 + (point[1] - obs_y)**2)
                safe_distance = distance - obs_radius
                
                if safe_distance < min_distance:
                    min_distance = safe_distance
                
                # Check for collision
                if safe_distance < 0:
                    return True, min_distance
        
        return False, min_distance
    
    def _record_violation(self, violation_type: str, distance: float):
        """Record safety violation."""
        self.violations.append({
            'type': violation_type,
            'distance': distance,
            'timestamp': time.time(),
            'robot_position': self.robot_status.position if self.robot_status else None
        })
    
    def get_stats(self) -> Dict[str, Any]:
        """Get safety monitor statistics."""
        return {
            'num_obstacles': len(self.obstacles),
            'num_violations': len(self.violations),
            'recent_violations': list(self.violations)[-5:] if self.violations else [],
            'min_safe_distance': self.min_safe_distance,
            'emergency_distance': self.emergency_distance,
        }

class CommandHandler:
    """High-level command handler for robot control."""
    
    def __init__(self, robot_comm: RobotCommunication):
        self.robot_comm = robot_comm
        self.command_registry = {}
        
        # Register default commands
        self._register_default_commands()
        
        logger.info("CommandHandler initialized")
    
    def _register_default_commands(self):
        """Register default commands."""
        self.register_command("move_forward", self._move_forward)
        self.register_command("move_backward", self._move_backward)
        self.register_command("rotate_left", self._rotate_left)
        self.register_command("rotate_right", self._rotate_right)
        self.register_command("stop", self._stop)
        self.register_command("follow_object", self._follow_object)
        self.register_command("avoid_obstacle", self._avoid_obstacle)
    
    def register_command(self, command_name: str, handler: callable):
        """Register a command handler."""
        self.command_registry[command_name] = handler
        logger.info(f"Command registered: {command_name}")
    
    def execute_command(self, command_name: str, **kwargs) -> bool:
        """
        Execute a registered command.
        
        Args:
            command_name: Name of the command to execute
            **kwargs: Arguments for the command handler
            
        Returns:
            True if command executed successfully
        """
        if command_name not in self.command_registry:
            logger.error(f"Unknown command: {command_name}")
            return False
        
        try:
            handler = self.command_registry[command_name]
            return handler(**kwargs)
        except Exception as e:
            logger.error(f"Error executing command {command_name}: {e}")
            return False
    
    def _move_forward(self, distance: float = 1.0, speed: float = 0.5) -> bool:
        """Move robot forward."""
        velocity = min(speed, self.robot_comm.config.max_linear_velocity)
        duration = distance / velocity if velocity > 0 else 0
        
        command = RobotCommand(
            command_type="move",
            linear_velocity=velocity,
            duration=duration
        )
        
        return self.robot_comm.command_queue.put_nowait(command)
    
    def _move_backward(self, distance: float = 1.0, speed: float = 0.5) -> bool:
        """Move robot backward."""
        return self._move_forward(distance=-distance, speed=speed)
    
    def _rotate_left(self, angle: float = 90.0, speed: float = 0.5) -> bool:
        """Rotate robot left."""
        angular_velocity = min(speed, self.robot_comm.config.max_angular_velocity)
        duration = np.radians(angle) / angular_velocity if angular_velocity > 0 else 0
        
        command = RobotCommand(
            command_type="rotate",
            angular_velocity=angular_velocity,
            duration=duration
        )
        
        return self.robot_comm.command_queue.put_nowait(command)
    
    def _rotate_right(self, angle: float = 90.0, speed: float = 0.5) -> bool:
        """Rotate robot right."""
        return self._rotate_left(angle=-angle, speed=speed)
    
    def _stop(self) -> bool:
        """Stop robot."""
        command = RobotCommand(
            command_type="stop"
        )
        
        return self.robot_comm.command_queue.put_nowait(command)
    
    def _follow_object(self, object_id: int, distance: float = 1.0) -> bool:
        """Follow a specific object."""
        # This would use object tracking and generate appropriate commands
        # For now, just move forward slowly
        return self._move_forward(distance=0.1, speed=0.2)
    
    def _avoid_obstacle(self, obstacle_position: Tuple[float, float]) -> bool:
        """Avoid a specific obstacle."""
        # Calculate avoidance direction
        robot_pos = self.robot_comm.status.position
        dx = obstacle_position[0] - robot_pos[0]
        dy = obstacle_position[1] - robot_pos[1]
        
        # Rotate away from obstacle
        if dx > 0:
            return self._rotate_left(angle=45)
        else:
            return self._rotate_right(angle=45)
    
    def handle_detection_commands(self, detections: List[DetectionCommand]) -> List[bool]:
        """
        Handle multiple detection commands.
        
        Args:
            detections: List of detection commands
            
        Returns:
            List of success flags for each command
        """
        results = []
        
        for detection in detections:
            # Convert detection to appropriate command
            if detection.action == "approach":
                success = self._move_forward(distance=0.5, speed=0.3)
            elif detection.action == "avoid":
                success = self._avoid_obstacle(detection.position[:2])
            elif detection.action == "track":
                success = self._follow_object(detection.detection_id)
            else:
                success = False
            
            results.append(success)
        
        return results