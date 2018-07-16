#!/usr/bin/env python

# Import modules
import numpy as np
import pickle
import sklearn
import time
import sensor_msgs.point_cloud2 as pc2
from sklearn.preprocessing import LabelEncoder
from sensor_msgs.msg import JointState
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker

import rospy
import tf
import yaml
from geometry_msgs.msg import Pose, Point
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter


# global variable for test scene num
TEST_SCENE_NUM = 1

# global variables for world joint rotation
HAS_TURNED_RIGHT = False
HAS_TURNED_LEFT = False
HAS_RETURNED_CENTER = False
POSITION_TOLERANCE = .005

# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster


# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict


# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)


# Helper function to get pr2 world joint position
def get_pr2_world_joint_position():
    pr2_joint_states = rospy.wait_for_message('/pr2/joint_states', JointState)
    return pr2_joint_states.position[-1]


# Helper function to rotate pr2 world joint
def rotate_pr2_world_joint():
    global HAS_TURNED_RIGHT
    global HAS_TURNED_LEFT
    global HAS_RETURNED_CENTER
    # get world joint position
    world_joint_position = get_pr2_world_joint_position()
    # establish  if pr2 world joint is facing right, left, or center
    facing_right = abs(world_joint_position + np.pi / 2) <= POSITION_TOLERANCE
    facing_left = abs(world_joint_position - np.pi / 2) <= POSITION_TOLERANCE
    facing_center = abs(world_joint_position) <= POSITION_TOLERANCE / 2
    # rotate world joint
    if not HAS_TURNED_RIGHT and not facing_right:
        pr2_world_joint_pub.publish(-np.pi / 2)
    elif not HAS_TURNED_RIGHT and facing_right:
        HAS_TURNED_RIGHT = True
        pr2_world_joint_pub.publish(np.pi / 2)
    elif not HAS_TURNED_LEFT and not facing_left:
        pr2_world_joint_pub.publish(np.pi / 2)
    elif not HAS_TURNED_LEFT and facing_left:
        HAS_TURNED_LEFT = True
        pr2_world_joint_pub.publish(0)
    elif not HAS_RETURNED_CENTER and not facing_center:
        pr2_world_joint_pub.publish(0)
    elif not HAS_RETURNED_CENTER and facing_center:
        HAS_RETURNED_CENTER = True


# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):
    # Exercise-2 TODOs:
    # Convert ROS msg to PCL data
    point_cloud = ros_to_pcl(pcl_msg)
    # Statistical Outlier Filtering
    outlier_filter = point_cloud.make_statistical_outlier_filter()
    outlier_filter.set_mean_k(50) # num of neighboring points to analyze
    x = .05 # threshold scale factor
    outlier_filter.set_std_dev_mul_thresh(x) # outlier > global_mean_distance + x*global_std_dev
    point_cloud_filtered = outlier_filter.filter()
    # Voxel Grid Downsampling
    vox = point_cloud_filtered.make_voxel_grid_filter()
    leaf_size = .005 # voxel (leaf) size
    vox.set_leaf_size(leaf_size, leaf_size, leaf_size)
    point_cloud_filtered = vox.filter()
    # PassThrough Filter
    passthrough = point_cloud_filtered.make_passthrough_filter()
    passthrough.set_filter_field_name('y')
    passthrough.set_filter_limits(-0.42, 0.42) # prevent bin recognition
    point_cloud_filtered = passthrough.filter()
    passthrough = point_cloud_filtered.make_passthrough_filter()
    passthrough.set_filter_field_name('z')
    passthrough.set_filter_limits(0.6, 1.8) # only table + objects
    point_cloud_filtered = passthrough.filter()
    # Moving Least Squares Smoothing (improved normal estimation)
    # mls = point_cloud_filtered.make_moving_least_squares()
    # mls.set_polynomial_fit(3)
    # mls.set_polynomial_order(True)
    # mls.set_search_radius(.01)
    # point_cloud_filtered = pcl.MovingLeastSquares_PointXYZRGB.process(mls)
    # RANSAC Plane Segmentation
    seg = point_cloud_filtered.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
    seg.set_distance_threshold(.01)
    inliers, coefficients = seg.segment()
    # Extract inliers and outliers
    cloud_table = point_cloud_filtered.extract(inliers, negative=False)
    cloud_objects = point_cloud_filtered.extract(inliers, negative=True)
    # Euclidean Clustering
    white_cloud = XYZRGB_to_XYZ(cloud_objects)
    tree = white_cloud.make_kdtree()
    ec = white_cloud.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(0.025) # distance tolerance
    ec.set_MinClusterSize(10)
    ec.set_MaxClusterSize(5500)
    ec.set_SearchMethod(tree)
    cluster_indices = ec.Extract() # extract indices for each of the discovered clusters
    # Create Cluster-Mask Point Cloud to visualize each cluster separately
    cluster_color = get_color_list(len(cluster_indices))
    color_cluster_point_list = []
    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([
                white_cloud[indice][0],
                white_cloud[indice][1],
                white_cloud[indice][2],
                rgb_to_float(cluster_color[j])
            ])
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)
    # Convert PCL data to ROS messages
    ros_cluster_cloud = pcl_to_ros(cluster_cloud)
    ros_cloud_objects = pcl_to_ros(cloud_objects)
    ros_cloud_table = pcl_to_ros(cloud_table)
    # Publish ROS messages
    pcl_cluster_pub.publish(ros_cluster_cloud)
    pcl_objects_pub.publish(ros_cloud_objects)
    pcl_table_pub.publish(ros_cloud_table)

    # Rotate PR2 in place to capture side tables for the collision map
    # if not HAS_RETURNED_CENTER:
    #     rotate_pr2_world_joint()
    #     return

    # Exercise-3 TODOs:
    detected_objects_labels = []
    detected_objects = []
    for index, pts_list in enumerate(cluster_indices):
        # Grab the points for the cluster from the extracted outliers (cloud_objects)
        pcl_cluster = cloud_objects.extract(pts_list)
        # convert the cluster from pcl to ROS using helper function
        ros_cluster = pcl_to_ros(pcl_cluster)
        # Extract histogram features
        # complete this step just as is covered in capture_features.py
        chists = compute_color_histograms(ros_cluster, using_hsv=True)
        normals = get_normals(ros_cluster)
        nhists = compute_normal_histograms(normals)
        feature = np.concatenate((chists, nhists))
        # Make the prediction, retrieve the label for the result
        # and add it to detected_objects_labels list
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)
        # Publish a label into RViz
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label,label_pos, index))
        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster
        detected_objects.append(do)
    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))
    # Publish the list of detected objects
    detected_objects_pub.publish(detected_objects)
    # move detected objects (if objects detected)
    if detected_objects:
        try:
            pr2_mover(detected_objects, ros_cloud_table)
        except rospy.ROSInterruptException:
            pass


# function to load parameters and request PickPlace service
def pr2_mover(detected_objects, ros_cloud_table):
    # Initialize variables
    yaml_dict_list, collision_map = [], []
    test_scene_num = Int32(data=TEST_SCENE_NUM)
    ros_cloud_table_data = [xyzrgb for xyzrgb in pc2.read_points(ros_cloud_table, 
        skip_nans=True, field_names=("x", "y", "z", "rgb"))]
    # Get/Read parameters
    listed_objects = rospy.get_param('/object_list')
    listed_dropboxes = rospy.get_param('/dropbox')
    # Parse parameters into individual variables
    listed_object_dict, listed_dropbox_dict = {}, {}
    for index in range(len(listed_objects)):
        listed_object_dict[listed_objects[index]['name']] = listed_objects[index]
    for index in range(len(listed_dropboxes)):
        listed_dropbox_dict[listed_dropboxes[index]['group']] = listed_dropboxes[index]

    # Loop through detected objects
    for index, detected_object in enumerate(detected_objects):
        # define object_name
        object_name = String(data=detected_object.label.tostring())
        matching_listed_object = listed_object_dict[detected_object.label]
        matching_dropbox = listed_dropbox_dict[matching_listed_object['group']]
        # define arm_name
        arm_name = String(data=matching_dropbox['name'])
        # establish pick_pose
        points_arr = ros_to_pcl(detected_object.cloud).to_array()
        centroid = np.mean(points_arr, axis=0)[:3].tolist()
        pick_pose = Pose(position=Point(*centroid))
        # establish place_pose
        random_x_shift = np.append(np.random.uniform(low=-.1, high=0), [0,0]) # prevent stacking
        shifted_dropbox_position = (matching_dropbox['position'] + random_x_shift).tolist()
        place_pose = Pose(position=Point(*shifted_dropbox_position))
        # establish collision map
        other_detected_objects = detected_objects[index + 1:]
        ros_cloud_other_objects_data = []
        if other_detected_objects:
            ros_cloud_other_objects_data = [[xyzrgb for xyzrgb in pc2.read_points(other_object.cloud, 
                skip_nans=True, field_names=("x", "y", "z", "rgb"))] for other_object in other_detected_objects]
            ros_cloud_other_objects_data = np.concatenate(ros_cloud_other_objects_data).tolist()
        collision_map_pcl_data = pcl.PointCloud_PointXYZRGB()
        collision_map_pcl_data.from_list(ros_cloud_table_data + ros_cloud_other_objects_data)
        collision_map = pcl_to_ros(collision_map_pcl_data)
        pcl_collision_map_pub.publish(collision_map)
        # append yaml dict
        yaml_dict = make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose)
        yaml_dict_list.append(yaml_dict)
        print '* Placing {} in {} dropbox at {}'.format(
            yaml_dict["object_name"], yaml_dict["arm_name"], yaml_dict["place_pose"])
        # place detected object in dropbox
        try:
            rospy.wait_for_service('pick_place_routine')
            pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)
            pick_place_routine(test_scene_num, object_name, arm_name, pick_pose, place_pose)
        except rospy.ServiceException, e:
            print "Service call failed: %s" % e
        # clear collision map
        print '* Clearing collision map'
        rospy.wait_for_service('/clear_octomap')
        clear_collision_map = rospy.ServiceProxy('/clear_octomap', Empty)
        clear_collision_map()
            
    # Output your request parameters into output yaml file
    yaml_filename = 'output_{}.yaml'.format(TEST_SCENE_NUM)
    print 'Writing {}'.format(yaml_filename)
    send_to_yaml(yaml_filename, yaml_dict_list)
    print 'Finished writing {}'.format(yaml_filename)

if __name__ == '__main__':
    # ROS node initialization
    rospy.init_node('clustering', anonymous=True)
    # Create Subscribers
    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)
    # Create Publishers
    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)
    pcl_cluster_pub = rospy.Publisher("/pcl_cluster", PointCloud2, queue_size=1)
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
    pcl_collision_map_pub = rospy.Publisher("/pr2/3d_map/points", PointCloud2, queue_size=1)
    pr2_world_joint_pub = rospy.Publisher('/pr2/world_joint_controller/command', Float64, queue_size=1)
    # Load Model From disk
    model = pickle.load(open('model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']
    # Initialize color_list
    get_color_list.color_list = []
    #  Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()
