import numpy as np
import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker


# Class used to deal with both model render functions (dominant stiffness and exact)
class CtmRender:
    def __init__(self):
        # Initialize node, subscribers and publishers
        rospy.init_node('ctm_env', anonymous=True)
        self.joints_pub = rospy.Publisher('ctm/command/joint', JointTrajectory, queue_size=10)
        self.ag_pub = rospy.Publisher('ctm/achieved_goal', Point, queue_size=10)
        self.dg_pub = rospy.Publisher('ctm/desired_goal', Point, queue_size=10)
        self.tube_backbone_pub = rospy.Publisher("tube_backbone_line", Marker, queue_size=100)

    def publish_achieved_goal(self, achieved_goal):
        ag_msg = Point()
        ag_msg.x = achieved_goal[0]
        ag_msg.y = achieved_goal[1]
        ag_msg.z = achieved_goal[2]
        self.ag_pub.publish(ag_msg)

    def publish_desired_goal(self, desired_goal):
        dg_msg = Point()
        dg_msg.x = desired_goal[0]
        dg_msg.y = desired_goal[1]
        dg_msg.z = desired_goal[2]
        self.dg_pub.publish(dg_msg)

    def publish_segments(self, segments):
        segment_points = []
        # TODO: Better way of doing this?
        for point in segments:
            seg_point = Point()
            seg_point.x = point[0] * 100
            seg_point.y = point[1] * 100
            seg_point.z = point[2] * 100
            segment_points.append(seg_point)

        # Set up markers
        marker = Marker()
        marker.header.frame_id = "world"
        marker.type = Marker.LINE_LIST
        marker.action = Marker.MODIFY
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.points = segment_points
        marker.pose.orientation.x = 0
        marker.pose.orientation.y = 0
        marker.pose.orientation.z = 0
        marker.pose.orientation.w = 1
        self.tube_backbone_pub.publish(marker)

    # Joints of the form [(beta+L)_0, ..., (beta+L)_n, alpha_0, ..., alpha_n]
    def publish_joints(self, joints):
        num_tubes = np.size(joints) / 2
        # Preprocessing of joints (need to flip before publishing for C++ code
        gamma = np.flip(joints[num_tubes:], axis=0)
        beta_L = joints[0:num_tubes]
        distal_length = np.ediff1d(np.flip(beta_L, axis=0), to_begin=beta_L[-1])
        joints = np.empty(num_tubes * 2, dtype=np.float)
        joints[0:num_tubes] = distal_length
        joints[num_tubes:] = gamma

        joint_point = JointTrajectoryPoint()
        joint_point.positions = joints
        joint_msg = JointTrajectory()
        joint_msg.points.append(joint_point)
        self.joints_pub.publish(joint_msg)