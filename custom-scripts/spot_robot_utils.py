import asyncio
import json
import math
import time
from xml.dom.expatbuilder import ParseEscape
import requests
from PIL import Image
import grpc
import uuid

from bosdyn.api.graph_nav import graph_nav_pb2, map_pb2, nav_pb2
from bosdyn.api import geometry_pb2
from bosdyn.api.spot_cam import audio_pb2, logging_pb2, camera_pb2, ptz_pb2
from bosdyn.client import create_standard_sdk, lease, estop, robot_command, robot_state, power
from bosdyn.client.time_sync import TimeSyncClient, TimeSyncEndpoint
from bosdyn.client.frame_helpers import get_odom_tform_body
from bosdyn.client.graph_nav import GraphNavClient
from bosdyn.client.spot_cam.audio import AudioClient
from bosdyn.client.spot_cam.lighting import LightingClient
from bosdyn.client.spot_cam.media_log import MediaLogClient
from bosdyn.client.spot_cam.ptz import PtzClient
from bosdyn.client.exceptions import ResponseError
from bosdyn.client import spot_cam
from bosdyn.client.estop import EstopKeepAlive
from bosdyn.client.lease import LeaseKeepAlive
from bosdyn.client.power import BatteryMissingError
from bosdyn.client.robot_command import RobotCommandBuilder
from bosdyn.client.docking import DockingClient
from bosdyn.geometry import EulerZXY

from bosdyn.client.power import *
from contextlib import contextmanager, suppress
import sys, os

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout


class Spot:
    """Creates a Spot robot object with corresponding services and functions."""

    def __init__(self, spot_name, spot_ip, user, pswd):
        """
        Instantiates the Spot class object and registers the required service
        clients. Takes as input the pre-defined robot credentials.

        :param spot_name: a robot name (e.g. "Doggo")
        :type spot_name: string

        :param spot_ip: Spot robot IP address
        :type spot_ip: string

        :param user: Spot user name 
        :type user: string

        :param pswd: Spot user password
        :type pswd: string

        :return: Spot robot object
        :rtype: Spot class instance
        """

        # Robot Credentials
        self.spot_name = spot_name
        self.spot_ip = spot_ip
        self.user = user
        self.pswd = pswd

        print("Initializing Spot Instance ...")

        # Robot SDK Initialization & Authentication
        self.sdk = create_standard_sdk(self.spot_name)
        spot_cam.register_all_service_clients(self.sdk)
        self.robot = self.sdk.create_robot(self.spot_ip)
        self.robot.authenticate(self.user, self.pswd)

        # Robot Time Sync
        self.robot.time_sync.wait_for_sync()
        self.time_sync_client = self.robot.ensure_client(TimeSyncClient.default_service_name)
        self.time_sync_endpoint = TimeSyncEndpoint(self.time_sync_client)
        time_sync_established = False
        while not time_sync_established:
            try:
                time_sync_established = self.time_sync_endpoint.establish_timesync()
            except ResponseError as err:
                print("Error while Syncing Robot Time : {}".format(err))
                break

        # Client & Service Registrations
        self.lease_client = self.robot.ensure_client(lease.LeaseClient.default_service_name)
        self.lease = None
        self.lease_keepalive = None

        self.estop_client = self.robot.ensure_client(estop.EstopClient.default_service_name)
        self.estop_endpoint = None
        self.estop_keepalive = None 

        self.power_client = self.robot.ensure_client(power.PowerClient.default_service_name)
        self.state_client = self.robot.ensure_client(robot_state.RobotStateClient.default_service_name)
        self.command_client = self.robot.ensure_client(robot_command.RobotCommandClient.default_service_name)
        self.graph_nav_client = self.robot.ensure_client(GraphNavClient.default_service_name)
        self.audio_client = self.robot.ensure_client(AudioClient.default_service_name)
        self.media_log_client = self.robot.ensure_client(MediaLogClient.default_service_name)
        self.ptz_client = self.robot.ensure_client(PtzClient.default_service_name)
        self.docking_client = self.robot.ensure_client(DockingClient.default_service_name)
        self.lighting_client = self.robot.ensure_client(LightingClient.default_service_name)
        self.se2velocityLimit = self.create_se2velocityLimit()

        # Empty GraphNav Graph Instantiations (Waypoints will be mapped here)
        self.current_graph = None
        self.current_edges = dict()  
        self.current_waypoint_snapshots = dict()  
        self.current_edge_snapshots = dict()  
        self.current_annotation_name_to_wp_id = dict()

        print("Spot SDK initialized successfully.")

    
    def power_off(self):
        """Powers off the robot."""
        print("Powering off robot ...")

        with suppress_stdout():
            self.take_over_lease()
        
        with suppress(Exception):
            safe_power_off_robot(self.command_client, self.state_client, self.power_client)

        print("Robot successfully powered off.")


    def get_battery_level(self):
        """
        Returns the robot's current battery level.

        :return: battery level in percent (%) 
        :rtype: float
        """
        return self.state_client.get_robot_state().battery_states[0].charge_percentage.value

    
    def take_over_lease(self):
        """Forcefully takes over the current robot lease."""

        if self.lease == None:
            print("Taking Lease ...")
            self.lease = self.lease_client.take()
            self.lease_keepalive = LeaseKeepAlive(self.lease_client)
            print("Lease acquired successfully.")
        else:
            print("Error : could not take over lease.")    


    def release_lease(self):
        """Releases the current robot lease."""

        if self.lease != None:
            print("Releasing Lease ...")
            self.lease_keepalive.shutdown()
            self.lease_client.return_lease(self.lease)
            self.lease = None
            print("Lease released successfully.")
        else:
            print("Error : could not release lease.")


    def create_se2velocityLimit(self, VELOCITY_BASE_SPEED = 0.2,
                                      VELOCITY_MAX_SPEED = 0.5, 
                                      VELOCITY_BASE_ANGULAR = 0.8,
                                      VELOCITY_MAX_ANGULAR = 0.8,
                                      VELOCITY_CMD_DURATION = 0.6):
        """
        Creates crucial velocity and robot movement parameters.

        :param VELOCITY_BASE_SPEED: base velocity speed (default=0.2)
        :type VELOCITY_BASE_SPEED: float
        
        :param VELOCITY_MAX_SPEED: maximum velocity speed (default=0.5)
        :type VELOCITY_MAX_SPEED: float

        :param VELOCITY_BASE_ANGULAR: base angular velocity (default=0.8)
        :type VELOCITY_BASE_ANGULAR: float

        :param VELOCITY_MAX_ANGULAR: maximum angular velocity (default=0.8)
        :type VELOCITY_MAX_ANGULAR: float

        :param VELOCITY_CMD_DURATION: duration of velocity command (default=0.6)
        :type VELOCITY_CMD_DURATION: float
        """
        min_linear = geometry_pb2.Vec2(x=VELOCITY_BASE_SPEED, y=VELOCITY_BASE_SPEED)
        max_linear = geometry_pb2.Vec2(x=VELOCITY_MAX_SPEED, y=VELOCITY_MAX_SPEED)
        min_vel = geometry_pb2.SE2Velocity(linear=min_linear, angular = 5.0)
        max_vel = geometry_pb2.SE2Velocity(linear=min_linear, angular = 5.0)
        self.se2velocityLimit = geometry_pb2.SE2VelocityLimit(min_vel=min_vel, max_vel=max_vel)


    def create_estop(self):
        """Creates an Estop."""

        if self.estop_endpoint == None:
            print("Creating an Estop Endoint ...")
            self.estop_endpoint = estop.EstopEndpoint(client = self.estop_client, 
                                                      name = self.spot_name + " ESTOP", 
                                                      estop_timeout = 0.1)
            self.estop_endpoint.force_simple_setup()
            self.estop_keepalive = EstopKeepAlive(self.estop_endpoint)
            print("Estop created successfully.")
        else:
            print("Error : could not create Estop.") 


    def delete_estop(self):
        """Deletes an Estop."""

        if self.estop_endpoint != None:
            print("Deleting Estop ...")
            self.estop_keepalive.stop()
            self.estop_keepalive = None
            self.estop_endpoint.stop()
            self.estop_endpoint = None
            print("Estop deleted successfully.")
        else:
            print("Error : could not delete Estop since there is none.")

    
    def enable_motor_power(self):
        """Enables the robot's powertrain."""

        self.take_over_lease()
        self.create_estop()

        try:
            print("Powering on motors ...")
            power.power_on(self.power_client)
            print("Motors are powered on.")
        except:
            print("Error : failed powering on motors.")


    def disable_motor_power(self):
        """Disables the robot's powertrain."""

        self.take_over_lease()
    
        try:
            print("Powering Off Motors ...")
            power.power_off(self.power_client)
            print("Motors are powered off.")
        except:
            print("Error : failed powering off motors.")

        self.release_lease()
        self.delete_estop()


    def play_audio(self, filename="autonomous_robot_en"):
        """
        Plays a pre-loaded audio file. 
        *Note* : The audio file needs to be a .wav file located in "spot-sdk/python/examples/spot_cam/data"
        
        :param filename: name of the .wav audio file (default="autonomous_robot_en")
        :type filename: string
        """
        sound = audio_pb2.Sound(name=filename)
        gain = 0.0
        self.audio_client.play_sound(sound, gain) 


    async def play_audio_async(self, filename="autonomous_robot_en"):
        """
        Async version of play_audio.

        Plays a pre-loaded audio file. 
        *Note* : The audio file needs to be a .wav file located in spot-sdk/python/examples/spot_cam/data
        
        :param filename: name of the .wav audio file (default="autonomous_robot_en")
        :type filename: string
        """
        sound = audio_pb2.Sound(name=filename)
        gain = 0.0
        self.audio_client.play_sound(sound, gain)
        await asyncio.sleep(10)   


    def dock(self):
        """Performs the docking process."""

        # Control parameter
        is_finished = False

        # Retrieve local time stamp in seconds
        epoch_time = int(time.time())

        # Completion time in seconds (default = 30)
        timestamp_local = epoch_time + 30 

        # Convert local time stamp to robot time using initialized time_sync_endpoint with established time_sync
        timestamp_robot = self.time_sync_endpoint.robot_timestamp_from_local_secs(timestamp_local)
        
        # Control sequence : issues the navigation command twice a second to ensure proper estop functionality 
        while not is_finished:
            try:
                dock_cmd_id = self.docking_client.docking_command(520, self.time_sync_endpoint.clock_identifier, timestamp_robot)
            except ResponseError as e:
                print("Error while trying to dock {}".format(e))
                break
            time.sleep(.5)  # Sleep for half a second to allow for command execution.
            is_finished = self.check_success_docking(dock_cmd_id)   # Feedback
            if is_finished:
                print("Finished Docking.")
                return 0


    async def dock_async(self, led_pattern="off"):
        """
        Async version of dock.
        
        Performs the docking process.
        
        :param led_pattern: indicates led_pattern ("off", "blinking", "fading") during dock process (default="off")
        :type led_pattern: string

        :return: True, False
        :rtype: boolean
        """

        # Configure LED pattern
        if led_pattern != "off":
            lighting_task = asyncio.create_task(self.set_led_pattern_async(led_pattern))
        else:
            pass

        # Control parameter
        is_finished = False

        # Retrieve local time stamp in seconds
        epoch_time = int(time.time())

        # Completion Time in seconds (default = 30)
        timestamp_local = epoch_time + 30 

        # Convert local time stamp to robot time using initialized time_sync_endpoint with established time_sync
        timestamp_robot = self.time_sync_endpoint.robot_timestamp_from_local_secs(timestamp_local)
        
        # Control sequence : issues the navigation command twice a second to ensure proper estop functionality 
        while not is_finished:
            try:
                dock_cmd_id = self.docking_client.docking_command(520, self.time_sync_endpoint.clock_identifier, timestamp_robot)
            except ResponseError as e:
                print("Error while trying to dock {}".format(e))
                break
            await asyncio.sleep(.5)  # Sleep for half a second to allow for command execution.
            is_finished = self.check_success_docking(dock_cmd_id)   # Feedback
            if is_finished:
                print("Finished Docking.")
                if led_pattern != "off":
                    lighting_task.cancel()
                else:
                    pass
                await self.set_led_brightness_async([0, 0, 0, 0])
        return 0


    def check_success_docking(self, command_id=-1):
        """
        Retrieves docking feedback from the robot via the passed command_id.

        :param command_id: feedback verification command (default = -1, i.e. no specific command status to be checked)
        :type command_id: int

        :return: True, False
        :rtype: boolean
        """ 

        if command_id == -1:
            return False
        else:
            pass

        # Get status
        status = self.docking_client.docking_command_feedback_full(command_id)

        if status == "STATUS_ERROR_DOCK_NOT_FOUND":
            print("Docking Error: dock not found")
            return True
        elif status == "STATUS_DOCKED":
            print("Docking Success: robot is docked")
            return True
        elif status == "STATUS_MISALIGNED":
            print("Docking Error: robot is misaligned")
            return True
        elif status == "STATUS_ERROR_COMMAND_TIMED_OUT":
            print("Docking Error: command timed out")
            return True
        elif status == "STATUS_IN_PROGRESS":
            print("Docking still in progress ...")
            return False
        elif status == "STATUS_ERROR_NOT_AVAILABLE":
            print("Docking Error : Dock not available.")
            return False
        else:
            return False    # Command not finished yet

    
    def return_home_and_dock(self, waypoint_number):
        """
        Returns the robot to a waypoint number in front of the docking station, 
        peforms the docking sequence and disables the motor power.
        
        :param waypoint_number: waypoint number which is nearest to the docking station 
        (optimally in front of it such that Spot can recognize the dock fiducial)
        :type waypoint_number: int
        """
        ### SUGGESTION : Keep here also, but keep everywhere else if PTZ angle is tampered with
        self.reset_PTZ_camera_position()
        self.navigate_to_waypoint(waypoint_number)
        self.dock()


    async def return_home_and_dock_async(self, waypoint_number, led_pattern='off'):
        """
        Async version of return_home_and_dock.

        Returns the robot to a waypoint number in front of the docking station, 
        peforms the docking sequence and disables the motor power.
        
        :param waypoint_number: waypoint number which is nearest to the docking station 
        (optimally in front of it such that Spot can recognize the dock fiducial)
        :type waypoint_number: int
        """
        # SAME SUGGESTION AS ABOVE
        await self.reset_PTZ_camera_position_async() 
        await self.navigate_to_waypoint_async(waypoint_number)
        await self.dock_async(led_pattern=led_pattern)
        await self.set_led_pattern_async("off")


    def set_PTZ_camera_position(self, pan, tilt, zoom, name="mech"):
        """"
        Sets the PTZ camera to a specified position.

        :param pan: pan angle in radians (between 0 and 360)
        :type pan: float 

        :param tilt: tilt angle in radians (between -30 and 90)
        :type tilt: float 

        :param zoom: zoom factor (between 1 and 30) 
        :type zoom: int

        :param name: name of the PTZ camera module
        :type name: string
        """
        self.ptz_client.set_ptz_position(ptz_pb2.PtzDescription(name=name), pan, tilt, zoom)


    async def set_PTZ_camera_position_async(self, pan, tilt, zoom, name="mech"):
        """"
        Async version of set_PTZ_camera_position.

        Sets the PTZ camera to a specified position.

        :param pan: pan angle in radians (between 0 and 360)
        :type pan: float 

        :param tilt: tilt angle in radians (between -30 and 90)
        :type tilt: float 

        :param zoom: zoom factor (between 1 and 30) 
        :type zoom: int

        :param name: name of the PTZ camera module
        :type name: string
        """
        self.ptz_client.set_ptz_position(ptz_pb2.PtzDescription(name=name), pan, tilt, zoom)
        await asyncio.sleep(0.1)


    def reset_PTZ_camera_position(self):
        """
        Resets the PTZ camera to the initial default position with parameters :
        pan=145.0, tilt=0.0, zoom=1.0
        """
        self.set_PTZ_camera_position(pan=145.0, tilt=0.0, zoom=1.0)
    

    async def reset_PTZ_camera_position_async(self):
        """
        Asnyc version of reset_PTZ_camera_position.

        Resets the PTZ camera to the initial default position with parameters :
        pan=145.0, tilt=0.0, zoom=1.0
        """
        await self.set_PTZ_camera_position_async(pan=145.0, tilt=0.0, zoom=1.0)


    def easy_ptz_async(self, easy_ptz_name=None, pan=90.0, tilt=30.0, zoom=2.0):
        """
        Wrapper function for some standard PTZ camera positions.

        :param easy_ptz_name: name of the PTZ position. Choose from :
        "pan_left", "pan_right", "tilt_down", "tilt_up", "zoom_in", "zoom_out"
        :type easy_ptz_name: string

        :param pan: pan camera angle in radians (between 0 and 360, default=90.0)
        :type pan: float 

        :param tilt: tilt camera angle in radians (between -30 and 90, default=30.0)
        :type tilt: float 

        :param zoom: zoom factor (between 1 and 30, default=2.0) 
        :type zoom: int
        """
        # Reference values
        pan_ref = 145.0
        tilt_ref = 0.0
        zoom_ref = 1.0

        if easy_ptz_name == "pan_left":
            pan = pan_ref - pan
        if easy_ptz_name == "pan_right":
            pan = pan_ref + pan
        if easy_ptz_name == "tilt_down":
            tilt = tilt_ref - tilt
        if easy_ptz_name == "tilt_up":
            tilt = tilt_ref + tilt
        if easy_ptz_name == "zoom_in":
            zoom = zoom
        if easy_ptz_name == "zoom_out":
            zoom = zoom_ref

        self.set_PTZ_camera_position_async(pan=pan, tilt=tilt, zoom=zoom)


    async def easy_ptz_async(self, easy_ptz_name=None, pan=90.0, tilt=30.0, zoom=2.0):
        """"
        Async version of easy_ptz.
        
        Wrapper function for some standard PTZ camera positions.

        :param easy_ptz_name: name of the PTZ position. Choose from :
        "pan_left", "pan_right", "tilt_down", "tilt_up", "zoom_in", "zoom_out"
        :type easy_ptz_name: string

        :param pan: pan camera angle in radians (between 0 and 360, default=90.0)
        :type pan: float 

        :param tilt: tilt camera angle in radians (between -30 and 90, default=30.0)
        :type tilt: float 

        :param zoom: zoom factor (between 1 and 30, default=2.0) 
        :type zoom: int    
        """
        # Reference values
        pan_ref = 145.0
        tilt_ref = 0.0
        zoom_ref = 1.0

        if easy_ptz_name == "pan_left":
            pan = pan_ref - pan
        if easy_ptz_name == "pan_right":
            pan = pan_ref + pan
        if easy_ptz_name == "tilt_down":
            tilt = tilt_ref - tilt
        if easy_ptz_name == "tilt_up":
            tilt = tilt_ref + tilt
        if easy_ptz_name == "zoom_in":
            zoom = zoom
        if easy_ptz_name == "zoom_out":
            zoom = zoom_ref

        await self.set_PTZ_camera_position_async(pan=pan, tilt=tilt, zoom=zoom)


    async def set_led_pattern_async(self, led_pattern="off"):
        """
        Async function to set a specific LED pattern such as "blinking" or "fading".
        Turns off LEDs if led_pattern = "off".

        :param led_pattern: name of the LED pattern (default="off")
        :type led_pattern: string
        """
        if led_pattern == "off":
            await self.set_led_brightness_async([0, 0, 0, 0])
        elif led_pattern == "fading":
            directions = [1, 1, 1, 1]
            brightness_values = [0.0, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
            idxs = [0, 3, 6, 9]
            brightnesses = [brightness_values[0], brightness_values[3], brightness_values[6], brightness_values[9]]
            while True:
                for idx, brightness in enumerate(brightnesses):
                    newIdx = idxs[idx] + directions[idx]
                    if newIdx >= len(brightness_values) - 1 or newIdx <= 0:
                        directions[idx] = directions[idx] * -1
                    idxs[idx] = newIdx
                    brightnesses[idx] = brightness_values[newIdx]
                self.lighting_client.set_led_brightness_async(brightnesses)
                await asyncio.sleep(0.1)
        elif led_pattern == "blinking":
            brightness_on = 1.0
            brightness_off = 0.0
            brightnesses_on = [brightness_on, brightness_on, brightness_on, brightness_on]
            brightnesses_off = [brightness_off, brightness_off, brightness_off, brightness_off]
            while True:
                self.lighting_client.set_led_brightness_async(brightnesses_on)
                await asyncio.sleep(0.3)
                self.lighting_client.set_led_brightness_async(brightnesses_off)
                await asyncio.sleep(0.3)
        else:
            print("Error : invalid LED pattern.")


    def set_led_brightness(self, brightnesses):
        """
        Configures the brightness setting for each LED. Expects an array of brightness
        values (from 0.0 to 1.0) for each of the 4 LEDs. 

        Example : brightnesses = [0.3 0.6 0.8 1.0] sets the following brightness values
        - LED_0 = 0.3
        - LED_1 = 0.6 
        - LED_2 = 0.8
        - LED_3 = 1.0 

        The LEDs are numerated from Spot Cam's left (LED_0) to right (LED_3) side.

        :param brightnesses: array of brightness values (from 0.0 to 1.0) 
        :type brightnesses:  array of floats
        """
        self.lighting_client.set_led_brightness(brightnesses)


    async def set_led_brightness_async(self, brightnesses):
        """"
        Async version of set_led_brightness.

        Configures the brightness setting for each LED. Expects an array of brightness
        values (from 0.0 to 1.0) for each of the 4 LEDs. 

        Example : brightnesses = [0.3 0.6 0.8 1.0] sets the following brightness values
        - LED_0 = 0.3
        - LED_1 = 0.6 
        - LED_2 = 0.8
        - LED_3 = 1.0 

        The LEDs are numerated from Spot Cam's left (LED_0) to right (LED_3) side.

        :param brightnesses: array of brightness values (from 0.0 to 1.0) 
        :type brightnesses:  array of floats

        :rtype: boolean
        """
        self.lighting_client.set_led_brightness(brightnesses)
        await asyncio.sleep(.01)
        return 0
    

    def robot_pose(self, yaw=0.0, roll=0.0, pitch=0.0):
        """
        Configures a specified 3-dimensional (yaw, roll, pitch) robot pose. 

        :param yaw: yaw angle in radians (default=0.0)
        :type yaw: float

        :param roll: roll angle in radians (default=0.0)
        :type roll: float

        :param pitch: pitch angle in radians (default=0.0)
        :type pitch: float
        """
        footprint_R_body = EulerZXY(yaw=yaw, roll=roll, pitch=pitch)
        command = RobotCommandBuilder.synchro_stand_command(footprint_R_body=footprint_R_body)
        self.execute_command(command)


    async def robot_pose_async(self, yaw=0.0, roll=0.0, pitch=0.0):
        """
        Async version of robot_pose.

        Configures a specified 3-dimensional (yaw, roll, pitch) robot pose. 

        :param yaw: yaw angle in radians (default=0.0)
        :type yaw: float

        :param roll: roll angle in radians (default=0.0)
        :type roll: float

        :param pitch: pitch angle in radians (default=0.0)
        :type pitch: float
        """
        footprint_R_body = EulerZXY(yaw=yaw, roll=roll, pitch=pitch)
        command = RobotCommandBuilder.synchro_stand_command(footprint_R_body=footprint_R_body)
        await self.execute_command_async(command)


    def easy_pose(self, easy_pose_name=None):
        """
        Wrapper function for some standard robot poses.

        :param easy_pose_name: name of the robot pose position. Choose from :
        "yaw_left", "yaw_right", "roll_right", "roll_left", "pitch_down", "pitch_up".
        :type easy_pose_name: string
        """

        yaw = 0.0
        roll = 0.0
        pitch= 0.0

        if easy_pose_name == "yaw_left":
            yaw = - 30.0
        if easy_pose_name == "yaw_right":
            yaw = + 30.0
        if easy_pose_name == "roll_right":
            roll = - 30.0
        if easy_pose_name == "roll_left":
            roll = + 30.0
        if easy_pose_name == "pitch_down":
            pitch = - 30.0
        if easy_pose_name == "pitch_up":
            pitch = + 30.0
        self.robot_pose_async(yaw=yaw, roll=roll, pitch=pitch)


    async def easy_pose_async(self, easy_pose_name=None):
        """
        Async version of easy_pose.

        Wrapper function for some standard robot poses.

        :param easy_pose_name: name of the robot pose position. Choose from :
        "yaw_left", "yaw_right", "roll_right", "roll_left", "pitch_down", "pitch_up".
        :type easy_pose_name: string
        """
        yaw = 0.0
        roll = 0.0
        pitch= 0.0

        if easy_pose_name == "yaw_left":
            yaw = - 30.0
        if easy_pose_name == "yaw_right":
            yaw = + 30.0
        if easy_pose_name == "roll_right":
            roll = - 30.0
        if easy_pose_name == "roll_left":
            roll = + 30.0
        if easy_pose_name == "pitch_down":
            pitch = - 30.0
        if easy_pose_name == "pitch_up":
            pitch = + 30.0
        await self.robot_pose_async(yaw=yaw, roll=roll, pitch=pitch)


    def execute_command(self, command):
        """
        Helper function which executes given robot commands.

        :param command: robot command (such as a robot pose)
        :type command: RobotCommandBuilder object
        """
        self.robot.start_time_sync()
        try:
            self.command_client.robot_command(command)
        except:
            print("COMMAND FAILED")


    async def execute_command_async(self, command):
        """
        Async version of execute_command.

        Helper function which executes given robot commands.

        :param command: robot command (such as a robot pose)
        :type command: RobotCommandBuilder object
        """
        self.robot.start_time_sync()
        try:
            await self.command_client.robot_command_async(command)
        except:
            print("COMMAND FAILED")


    def navigate_to_waypoint(self, waypoint_number):
        """
        Navigates the robot to a specified waypoint.

        :param waypoint_number: desired waypoint number
        :type waypoint_number: int

        :return: waypoint_number
        :rtype: int
        """
        # Retrieve the unique waypoint id and set the necessary travel parameters
        waypoint_id = self.find_unique_waypoint_id(waypoint_number=waypoint_number, graph=self.current_graph, name_to_id=self.waypoint_name_to_id)
        travel_params = self.graph_nav_client.generate_travel_params(0.0, 0.0, self.se2velocityLimit)
        print("navigating to waypoint ", waypoint_number, " with id: ", waypoint_id)

        # Navigation process
        is_finished = False
        while not is_finished:
            try:
                nav_to_cmd_id = self.graph_nav_client.navigate_to(waypoint_id, 1.0, travel_params=travel_params)
            except ResponseError as e:
                print("Error while navigating {}".format(e))
                break
            time.sleep(.5)  # Sleep for half a second to allow for command execution.
            is_finished = self.check_success_navigation(nav_to_cmd_id)
        return waypoint_number


    async def navigate_to_waypoint_async(self, waypoint_number):
        """
        Async version of navigate_to_waypoint.

        Navigates the robot to a specified waypoint.

        :param waypoint_number: desired waypoint number
        :type waypoint_number: int

        :return: waypoint_number
        :rtype: int
        """
        # Retrieve the unique waypoint id and set the necessary travel parameters
        waypoint_id = self.find_unique_waypoint_id(waypoint_number=waypoint_number, graph=self.current_graph, name_to_id=self.waypoint_name_to_id)
        travel_params = self.graph_nav_client.generate_travel_params(0.0, 0.0, self.se2velocityLimit)
        print("navigating to waypoint ", waypoint_number, " with id: ", waypoint_id)

        # Navigation process
        is_finished = False
        while not is_finished:
            try:
                nav_to_cmd_id = self.graph_nav_client.navigate_to(waypoint_id, 1.0, travel_params=travel_params)
            except ResponseError as e:
                print("Error while navigating {}".format(e))
                break
            await asyncio.sleep(0.5)  # Sleep for half a second to allow for command execution.
            is_finished = self.check_success_navigation(nav_to_cmd_id)
            # if is_finished:
            #     print("navigation to", waypoint_number, " finished")
        return waypoint_number


    def check_success_navigation(self, command_id=-1):
        """
        Retrieves navigation feedback from the robot via the passed command_id.

        :param command_id: feedback verification command (default = -1, i.e. no specific command status to be checked)
        :type command_id: int

        :return: True, False
        :rtype: boolean
        """ 

        if command_id == -1:
            return False 

        # Get status
        status = self.graph_nav_client.navigation_feedback(command_id)

        if status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_REACHED_GOAL:
            return True
        elif status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_LOST:
            print("Robot got lost when navigating the route, the robot will now sit down.")
            return True
        elif status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_STUCK:
            print("Robot got stuck when navigating the route, the robot will now sit down.")
            return True
        elif status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_ROBOT_IMPAIRED:
            print("Robot is impaired.")
            return True
        else:
            return False    # Command not finished yet


    def id_to_short_code(self, id):
        """
        Converts a unique waypoint id to a 2 letter short code (if possible, else returns 0).

        :param id: unique waypoint id
        :type id: string

        :return: 2 letter short code
        :rtype: string
        """
        tokens = id.split('-')
        if len(tokens) > 2:
            return '%c%c' % (tokens[0][0], tokens[1][0])
        return None


    def find_unique_waypoint_id(self, waypoint_number, graph, name_to_id):
        """
        Returns the unique waypoint id of a specified waypoint_number.

        :param waypoint_number: waypoint number
        :type waypoint_number: int

        :param graph: graph associated with the given waypoint number
        :type graph: robot graph object (default=self.current_graph)

        :param name_to_id: dictionary containing unique waypoint names and ids
        :type name_to_id: dict

        :return: unique waypoint id
        :rtype: string
        """

        # Check graph
        if graph is None:
            print("Error : waypoints cannot be listed and converted since there is no graph.")
            return 

        # Distinguish between waypoint numbers and name strings
        if isinstance(waypoint_number, int):
            waypoint_name = "waypoint_{}".format(waypoint_number)
        elif isinstance(waypoint_number, str):
            waypoint_name = waypoint_number

        # Waypoint conversion
        if len(waypoint_name) != 2:
            if waypoint_name in name_to_id:
                if name_to_id[waypoint_name] is not None:
                    return name_to_id[waypoint_name]
                else:
                    print("The waypoint name %s is used for multiple different unique waypoints. Please use the unique waypoint id." % (waypoint_name))
                    return None
            return waypoint_name

        ret = waypoint_name
        for waypoint in graph.waypoints:
            if waypoint_name == self.id_to_short_code(waypoint.id):
                if ret != waypoint_name:
                    return waypoint_name 
                ret = waypoint.id
        return ret


    def update_waypoints_and_edges(self, graph):
        """
        Provides and update to the waypoint and edge ids and returns corresponding dictionaries.
        
        :param graph: associated graph
        :type graph: robot graph object (default=self.current_graph)

        :return name_to_id: dictionary containing waypoint ids 
        :rtype name_to_id: dict

        :return edges: dictionary containing edge ids
        :rtype edges: dict
        """

        name_to_id = dict()
        edges = dict()
        short_code_to_count = {}
        waypoint_to_timestamp = []

        for waypoint in graph.waypoints:
            timestamp = -1.0
            try:
                timestamp = waypoint.annotations.creation_time.seconds + waypoint.annotations.creation_time.nanos / 1e9
            except:
                pass
            waypoint_to_timestamp.append((waypoint.id, timestamp, waypoint.annotations.name))

            # Determine how many waypoints have the same short code.
            short_code = self.id_to_short_code(waypoint.id)
            if short_code not in short_code_to_count:
                short_code_to_count[short_code] = 0
            short_code_to_count[short_code] += 1

            # Add the annotation name/id into the current dictionary.
            waypoint_name = waypoint.annotations.name
            if waypoint_name:
                if waypoint_name in name_to_id:
                    # Waypoint name is used for multiple different waypoints, so set the waypoint id
                    # in this dictionary to None to avoid confusion between two different waypoints.
                    name_to_id[waypoint_name] = None
                else:
                    # First time we have seen this waypoint annotation name. Add it into the dictionary
                    # with the respective waypoint unique id.
                    name_to_id[waypoint_name] = waypoint.id

        # Sort the set of waypoints by their creation timestamp. If the creation timestamp is unavailable,
        # fallback to sorting by annotation name.
        waypoint_to_timestamp = sorted(waypoint_to_timestamp, key=lambda x: (x[1], x[2]))

        for edge in graph.edges:
            if edge.id.to_waypoint in edges:
                if edge.id.from_waypoint not in edges[edge.id.to_waypoint]:
                    edges[edge.id.to_waypoint].append(edge.id.from_waypoint)
            else:
                edges[edge.id.to_waypoint] = [edge.id.from_waypoint]

        return name_to_id, edges


    async def navigate_to_waypoint_with_action_async(self, waypoint_number, easy_pose_name=None, pose=None, pose_duration=2.0, easy_ptz_name=None, ptz=None, led_pattern="fading", audio="on", camera=None, cloud_upload=False):
        """
        Asnyc function.

        Navigates the robot to a specified waypoint and executes a specified action upon arrival.

        :param waypoint_number: waypoint number
        :type waypoint_number: int

        :param easy_pose_name: name of the "easy_pose" to be executed
        Choose from : "yaw_left", "yaw_right", "roll_left", "roll_right", "pitch_down", "pitch_up"
        :type easy_pose_name: string

        :param pose: dictionary of yaw, pitch, roll poses for a customized action 
        :type pose: dict containing the keys "yaw", "pitch", "roll"

        :param pose_duration: duration of the specified pose in seconds
        :type pose_duration: float

        :param easy_ptz_name: name of the "easy_ptz" camera position
        Choose from : "pan_left", "pan_right", "tilt_down", "tilt_up", "zoom_in", "zoom_out"
        :type easy_ptz_name: string

        :param ptz: dictionary of pan, tilt, zoom values for a customized PTZ position
        :type ptz: dict containing the keys "pan", "tilt", "zoom"

        :param led_pattern: led pattern such as "blinking", "fading", "off"
        :type led_pattern: string

        :param audio: "on"/"off" parameter to allow for audio playback during navigation 
        :type audio: string
        """
        lighting_task = asyncio.create_task(self.set_led_pattern_async(led_pattern))

        if audio == "on":
            audio_task = asyncio.create_task(self.play_audio_async())
        elif audio == "off":
            pass

        await self.navigate_to_waypoint_async(waypoint_number=waypoint_number)
        tasks = set(asyncio.all_tasks()) - set([asyncio.current_task()])
        for t in tasks:
            t.cancel()

        # print("switching leds off after reaching waypoint ", waypoint_number)

        await self.set_led_pattern_async()

        if easy_pose_name:
            f1 = asyncio.create_task(self.easy_pose_async(easy_pose_name=easy_pose_name))
        elif pose:
            f1 = asyncio.create_task(self.robot_pose_async(yaw=pose.yaw, roll=pose.roll, pitch=pose.pitch))
        
        if easy_ptz_name:
            f2 = asyncio.create_task(self.easy_ptz_async(easy_ptz_name=easy_ptz_name))
        elif ptz:
            f2 = asyncio.create_task(self.set_PTZ_camera_position_async(pan=ptz.pan.value, tilt=ptz.tilt.value, zoom=ptz.zoom.value))

        await asyncio.sleep(pose_duration / 2)
        logpoint = None
        if camera is not None:
            logpoint = self.take_image(camera_name=camera)
        
        if logpoint is not None and cloud_upload == True:
            self.upload_image_to_IBM_COS_bucket(logpoint, waypoint_number)
        
        await asyncio.sleep(pose_duration / 2)

        await self.reset_PTZ_camera_position_async()


    def clear_graph(self):
        """"Clears the current GraphNav graph."""
        self.graph_nav_client.clear_graph()


    def download_graph(self):
        """Downloads the current GraphNav graph to the file directory."""
        self.graph_nav_client.download_graph()


    def upload_graph(self, file_path, forceful=False):
        """"Uploads a specified GraphNav graph to the robot. To prevent numerous 
        unnecessary re-uploads during testing and debugging, the function compares 
        the number of waypoints on the current robot graph vs. the number of waypoints
        on the specified upload graph. In case of a match, no re-upload is performed. 
        Alternatively, a re-upload can be forced via the *force* variable.

        :param file_path: absolute path to the graph
        :type file_path: string

        :param forceful: forces a graph (re-)upload
        :type forceful: boolean
        """

        # Load graph structure
        with open(file_path + "/graph", "rb") as graph_file:
            data = graph_file.read()
            self.new_graph = map_pb2.Graph()
            self.new_graph.ParseFromString(data)
            number_of_new_waypoints = len(self.new_graph.waypoints)

            # Get current robot graph
            self.current_graph = self.graph_nav_client.download_graph()
            number_of_current_waypoints = len(self.current_graph.waypoints)

            # Determine re-upload
            new_graph_upload_needed = True
            if number_of_current_waypoints == number_of_new_waypoints and forceful == False:
                new_graph_upload_needed = False
                print("Re-upload skipped. Graphs are identical.")
            else:
                print("Loaded graph has {} waypoints and {} edges".format(
                    len(self.new_graph.waypoints), len(self.new_graph.edges)))
                self.clear_graph()
                self.current_graph = self.new_graph

        # Upload Process
        if new_graph_upload_needed == True:

            # Load waypoints
            for waypoint in self.current_graph.waypoints:
                with open(file_path + "/waypoint_snapshots/{}".format(waypoint.snapshot_id),
                        "rb") as snapshot_file:
                    waypoint_snapshot = map_pb2.WaypointSnapshot()
                    waypoint_snapshot.ParseFromString(snapshot_file.read())
                    self.current_waypoint_snapshots[waypoint_snapshot.id] = waypoint_snapshot

            # Load edges
            for edge in self.current_graph.edges:
                if len(edge.snapshot_id) == 0:
                    continue
                else:
                    print("Error while uploading graph : edge.snapshot_id has nonzero length.")

                with open(file_path + "/edge_snapshots/{}".format(edge.snapshot_id),
                        "rb") as snapshot_file:
                    edge_snapshot = map_pb2.EdgeSnapshot()
                    edge_snapshot.ParseFromString(snapshot_file.read())
                    self.current_edge_snapshots[edge_snapshot.id] = edge_snapshot

            # Upload graph structure
            if len(self.current_graph.anchoring.anchors) == 0:
                do_anchoring = True
            else:
                do_anchoring = False

            response = self.graph_nav_client.upload_graph(lease=self.lease.lease_proto,
                                                        graph=self.current_graph,
                                                        generate_new_anchoring=do_anchoring)

            # Upload waypoint and edge snapshots
            for snapshot_id in response.unknown_waypoint_snapshot_ids:
                waypoint_snapshot = self.current_waypoint_snapshots[snapshot_id]
                self.graph_nav_client.upload_waypoint_snapshot(waypoint_snapshot)

            for snapshot_id in response.unknown_edge_snapshot_ids:
                edge_snapshot = self.current_edge_snapshots[snapshot_id]
                self.graph_nav_client.upload_edge_snapshot(edge_snapshot)

        else:
            pass

        # Check for initial localization
        localization_state = self.graph_nav_client.get_localization_state()
        if not localization_state.localization.waypoint_id:
            print("Upload complete. Robot is NOT initialized.")
        else:
            print("Upload complete. Robot IS initialized.")


    def set_initial_localization_waypoint(self, waypoint_number=None, dock_waypoint_number=0, forceful=False):
        """
        Initializes the robot corresponding to the specified localization waypoint. 

        In case that the robot is docked, you can only initialize the waypoint with a *forceful* statement. 
        This is especially useful when the position of the docking station has changed.

        :param waypoint_number: initializing waypoint number
        :type waypoint_number: int

        :param dock_waypoint_number: waypoint number associated with the docking station (default=0)
        :type dock_waypoint_number: int

        :param forceful: determines forceful override in the rare case that the docking station has changed its original position
        :type forceful: boolean
        """
        
        # Docking station comparison
        docking_state = self.docking_client.get_docking_state()

        if docking_state.status == 1:
            if waypoint_number is not None and forceful == True:
                    init_waypoint_number = waypoint_number
            else:
                init_waypoint_number = dock_waypoint_number
        else:
            if waypoint_number is not None and forceful == True:
                    init_waypoint_number = waypoint_number
            else:
                pass
        
        # Retrieve the destination waypoint via the specified waypoint number
        self.waypoint_name_to_id, self.edges = self.update_waypoints_and_edges(self.current_graph)
        destination_waypoint = self.find_unique_waypoint_id(waypoint_number=init_waypoint_number, 
                                                            graph=self.current_graph, 
                                                            name_to_id=self.waypoint_name_to_id)
        
        if not destination_waypoint:
            return
        else:
            pass

        # Retrieve the current odometry status
        robot_state = self.state_client.get_robot_state()
        current_odom_tform_body = get_odom_tform_body(
            robot_state.kinematic_state.transforms_snapshot).to_proto()
        
        # Set the initial localization as specified by the waypoint
        localization = nav_pb2.Localization()
        localization.waypoint_id = destination_waypoint
        localization.waypoint_tform_body.rotation.w = 1.0
        self.graph_nav_client.set_localization(
            initial_guess_localization=localization,
            # It's hard to get the pose perfect, search +/-20 deg and +/-20cm (0.2m).
            max_distance=0.2,
            max_yaw=20.0 * math.pi / 180.0,
            fiducial_init=graph_nav_pb2.SetLocalizationRequest.FIDUCIAL_INIT_NO_FIDUCIAL,
            ko_tform_body=current_odom_tform_body)

        print("Localization to waypoint {} initialized successfully.".format(init_waypoint_number))


    def clear_and_upload_graph(self, file_path, init_waypoint_number=None, forceful=False):
        """
        Clears the current graph on the robot and uploads a new one. Similarly to upload_graph, 
        a forceful upload can be triggered. If not, the function compares the current and new
        graph and decides to not upload in case they both match.
        
        :param file_path: absolute path to the graph
        :type file_path: string

        :param init_waypoint_number: initializing waypoint number
        :type init_waypoint_number: int

        :param forceful: forces a graph (re-)upload (default=False)
        :type forceful: boolean
        """

        # Download the current graph
        self.current_graph = self.graph_nav_client.download_graph()

        # Trigger the upload_graph procedure
        self.upload_graph(file_path, forceful=forceful)

        if init_waypoint_number is not None:
            self.set_initial_localization_waypoint(init_waypoint_number)
        else:
            pass


    def take_image(self, camera_name="ptz"):
        """
        Takes and image using the specified camera and saves it to the local directory.
        
        :param camera_name: name of the camera
        :type camera_name: string
        """
        args = (camera_pb2.Camera(name=camera_name), logging_pb2.Logpoint.STILLIMAGE)
        logpoint = self.media_log_client.store(*args)
        print("Image saved.")
        return logpoint


    async def upload_image_to_IBM_COS_bucket_async(self, logpoint, waypoint_number):
        # IBM COS bucket data
        url_to_IBM_COS_bucket = "https://s3.eu-de.cloud-object-storage.appdomain.cloud/sapai-spot/"
        filename_prefix = "image_at_waypoint_"
        filename_extension = ".jpg"

        # image is a tupel of length 2: first part is log point meta data, second part is corresponding image data
        image_raw_data = MediaLogClient.retrieve_raw_data(self.media_log_client, logpoint)
        imgage = Image.frombytes("RGB", (1920, 1080), image_raw_data[1], "raw")
        # imgage = Image.frombuffer("L", (1920, 1080), image_raw_data[1], "raw")
        # imgage.show()
        imgage.save("temp.jpg")
        with open("temp.jpg", "rb") as f:
            data = f.read() #.replace(b'\n', b'')
        # Get IBM Cloud access token - valid for 1h
        bearer = self.get_bearer_access_token()
        # Uploading image to IBM Cloud
        print("uploading image...")
        headers2 = {
            'Content-Type': 'image/jpg',
            'Authorization': bearer,
        }
        full_filename_url = url_to_IBM_COS_bucket + filename_prefix + str(waypoint_number) + filename_extension
        response2 = requests.put(full_filename_url, headers=headers2, data=data)
        await asyncio.sleep(15)
        pprint(response2)
        print("upload finished")

    def get_bearer_access_token(self):
        with open("apikey.txt", "r") as file:
            apikey = file.read().replace("\n", "")
        
        print(apikey)
        print("getting token...")
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
        }
        # data_token = "grant_type=urn:ibm:params:oauth:grant-type:apikey&apikey=a5pV8D7c8K-eq5PFVV_ri1Zu1FYXlCsU0eMoNEPQ1dbT"
        data_token = "grant_type=urn:ibm:params:oauth:grant-type:apikey&apikey=" + apikey
        response = requests.post("https://iam.cloud.ibm.com/identity/token", headers=headers, data=data_token)
        
        r0 = vars(response)
        c0 = json.loads(r0["_content"])
        access_token = c0["access_token"]
        bearer = 'Bearer ' + access_token
        # pprint(bearer)
        print("token recieved")
        return bearer

    def take_photo_new(self, camera_name="ptz"):
        """
        Takes an image using the specified camera and saves it to the local directory.
        
        :param camera_name: name of the camera
        :type camera_name: string
        """
        args = (camera_pb2.Camera(name=camera_name), logging_pb2.Logpoint.STILLIMAGE)
        logpoint = self.media_log_client.store(*args)
        image_raw_data = MediaLogClient.retrieve_raw_data(self.media_log_client, logpoint)
        image = Image.frombytes("RGB", (1920, 1080), image_raw_data[1], "raw")
        # Save to a unique filename and return its path so callers can manage it
        filename = str(uuid.uuid4()) + ".jpg"
        image.save(filename)
        print("SAVED.")
        return filename
        

# async def main_async(): 
#     # Credentials
#     user = "spot-dev"
#     pswd = "Watson2201Center!"
#     spot_ip = "172.30.28.201"
#     spot_name = "Pluto"
# 
#     # Robot Object Instantiation
#     spot = Spot(spot_name, spot_ip, user, pswd)
#     print("BATTERY LEVEL : {} %".format(spot.get_battery_level()))
# 
#     # Enable Power, Clear and Upload Graph, Initialize to waypoint 0
#     spot.enable_motor_power()
#     spot.clear_and_upload_graph(file_path="/Users/aladindjuhera/Projects/spot/spot/graphs/graph6", init_waypoint_number=0)
# 
#     # Perform Actions
#     await spot.navigate_to_waypoint_with_action_async(waypoint_number=2, easy_pose_name="roll_left", easy_ptz_name="tilt_down", led_pattern="fading", audio="yes")
#     await spot.return_home_and_dock_async(waypoint_number=147, led_pattern="off")
# 
# 
# if __name__ == '__main__':
#     asyncio.run(main_async())
