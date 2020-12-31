import numpy as np
from scipy.spatial.transform import Rotation as R

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Class used to deal with both model render functions (dominant stiffness and exact)
class CtmRender:
    def __init__(self, model, tubes):
        self.model = model
        self.tubes = tubes
        self.k = [i.k for i in self.tubes]
        self.l_curved = [i.L_c for i in self.tubes]
        self.tube_lengths = [i.L for i in self.tubes]

    def publish_transforms(self, transforms):
        nPts = len(transforms)
        # marker array allocation
        marker_array = MarkerArray()
        header_stamp = rospy.Time.now()

        for j, backbonePt in enumerate(transforms):
            marker = Marker()
            marker.header.stamp = header_stamp
            marker.header.frame_id = "/world"
            marker.type = Marker.CYLINDER
            marker.color.a = 1.0

            marker.id = j
            px = backbonePt[0, 3]
            py = backbonePt[1, 3]
            pz = backbonePt[2, 3]

            marker.action = Marker.ADD

            # Compute gap
            gap = 0
            if j < (nPts - 1):
                px_next = transforms[j + 1][0, 3]
                py_next = transforms[j + 1][1, 3]
                pz_next = transforms[j + 1][2, 3]
                gap = np.sqrt(
                    (px - px_next) * (px - px_next) + (py - py_next) * (py - py_next) + (pz - pz_next) * (pz - pz_next))

            if gap < 0.00001:
                gap = 0.00001

            pr = R.from_dcm(backbonePt[:3, :3])
            qx = pr.as_quat()[0]
            qy = pr.as_quat()[1]
            qz = pr.as_quat()[2]
            qw = pr.as_quat()[3]

            if j in [0,9,19]:
                # basically (0, 0, 1) rotated by the quaternion then times half the gap is the displaced amount
                marker.pose.position.x = px+gap / 2 * (2 * qw * qy-2 * qz * qx)
                marker.pose.position.y = py+gap / 2 * (2 * qw * qx+2 * qy * qz)
                marker.pose.position.z = pz+gap / 2 * (1+2 * qx * qx-2 * qy * qy)
            else:
                marker.pose.position.x = px
                marker.pose.position.y = py
                marker.pose.position.z = pz

            marker.pose.orientation.x = pr.as_quat()[0]
            marker.pose.orientation.y = pr.as_quat()[1]
            marker.pose.orientation.z = pr.as_quat()[2]
            marker.pose.orientation.w = pr.as_quat()[3]

            # generally make the length of the cylinders twice the gaps
            # For the first cylinder, length should equal gap
            # Length of last marker arbitrarily small

            if j == nPts:
                marker.scale.z = 0.00000005
            elif j in [0,9,19]:
                marker.scale.z = gap
            else:
                marker.scale.z = gap * 2

            # For now set same color. TODO: separate color per tube with correct radius
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.scale.x = 1.168e-3
            marker.scale.y = 1.168e-3

            marker_array.markers.append(marker)

        # Publish marker array
        self.viz_pub.publish(marker_array)

    def publish_segments(self, segments):
        # Publish a marker array of cylinders (TODO: add in colour for different tubes)
        nPts = len(segments)

        # marker array allocation
        marker_array = MarkerArray()

        for j, backbonePt in enumerate(segments):
            marker = Marker()
            marker.header.stamp = rospy.Time.now()
            marker.header.frame_id = "/world"
            marker.type = Marker.CYLINDER
            marker.color.a = 1.0

            marker.id = j
            px = backbonePt[0]
            py = backbonePt[1]
            pz = backbonePt[2]
            # TODO: Add quaternion

            marker.action = Marker.MODIFY

            # Compute gap
            gap = 0
            if j < (nPts - 1):
                px_next = segments[j + 1][0]
                py_next = segments[j + 1][1]
                pz_next = segments[j + 1][2]
                gap = np.sqrt(
                    (px - px_next) * (px - px_next) + (py - py_next) * (py - py_next) + (pz - pz_next) * (pz - pz_next))

            if gap < 0.00001:
                gap = 0.00001

            marker.pose.position.x = px
            marker.pose.position.y = py
            marker.pose.position.z = pz

            # TODO: Correctly set orientation
            marker.pose.orientation.x = 0
            marker.pose.orientation.y = 0
            marker.pose.orientation.z = 0
            marker.pose.orientation.w = 1

            # generally make the length of the cylinders twice the gaps
            # For the first cylinder, length should equal gap
            # Length of last marker arbitrarily small

            if j == nPts:
                marker.scale.z = 0.00000005
            elif j == 1:
                marker.scale.z = gap
            else:
                marker.scale.z = gap * 2

            # For now set same color. TODO: separate color per tube with correct radius
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.scale.x = 1.168e-3
            marker.scale.y = 1.168e-3

            marker_array.markers.append(marker)

        # Publish marker array
        self.viz_pub.publish(marker_array)

        """
        segment_points = []
        # TODO: Better way of doing this?
        for point in segments:
            seg_point = Point()
            seg_point.x = point[0] * self.scale_factor
            seg_point.y = point[1] * self.scale_factor
            seg_point.z = point[2] * self.scale_factor
            segment_points.append(seg_point)

        # Set up markers
        marker = Marker()
        marker.header.frame_id = "world"
        marker.type = Marker.POINTS
        marker.action = Marker.MODIFY
        marker.scale.x = 1e-3
        marker.scale.y = 1e-3
        marker.scale.z = 1e-3
        marker.color.a = 1
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.points = segment_points
        marker.pose.orientation.x = 0
        marker.pose.orientation.y = 0
        marker.pose.orientation.z = 0
        marker.pose.orientation.w = 1
        self.tube_backbone_pub.publish(marker)
        """
