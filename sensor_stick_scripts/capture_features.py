#!/usr/bin/env python
import numpy as np
import pickle
import rospy
import time

from gazebo_msgs.srv import GetModelState
from sensor_stick.pcl_helper import *
from sensor_stick.training_helper import spawn_model
from sensor_stick.training_helper import delete_model
from sensor_stick.training_helper import initial_setup
from sensor_stick.training_helper import capture_sample
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from sensor_stick.srv import GetNormals
from geometry_msgs.msg import Pose
from sensor_msgs.msg import PointCloud2


def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster


def get_model_state():
    get_model_state_prox = rospy.ServiceProxy('gazebo/get_model_state', GetModelState)
    model_state = get_model_state_prox('training_model', 'world')
    print 'model_state.pose:', model_state.pose
    print 'model_state.success:', model_state.success
    print 'model_state.status_message:', model_state.status_message


if __name__ == '__main__':
    rospy.init_node('capture_node')
    models = [
       'sticky_notes',
       'book',
       'snacks',
       'biscuits',
       'eraser',
       'soap2',
       'soap',
       'glue']
    # Disable gravity and delete the ground plane
    initial_setup()
    labeled_features = []

    for model_name in models:
        spawn_model(model_name)

        for i in range(300):
            print '* Model|iteration: {} ({})'.format(model_name, i)
            sample_was_good = False
            try_count = 0
            while not sample_was_good and try_count < 20:
                sample_cloud = capture_sample()
                sample_cloud_arr = ros_to_pcl(sample_cloud).to_array()

                # Check for invalid clouds.
                if sample_cloud_arr.shape[0] == 0:
                    print('Invalid cloud detected')
                    try_count += 1
                else:
                    sample_was_good = True

            # Extract histogram features
            chists = compute_color_histograms(sample_cloud, using_hsv=True)
            normals = get_normals(sample_cloud)
            nhists = compute_normal_histograms(normals)
            feature = np.concatenate((chists, nhists))
            labeled_features.append([feature, model_name])
        try:
            get_model_state()
            delete_model()
            get_model_state()
        except Exception as e:
            print 'error:', e.message
        try:
            delete_model()
            get_model_state()
        except Exception as e:
            print 'error:', e.message
        try:
            delete_model()
            get_model_state()
        except Exception as e:
            print 'error:', e.message
        try:
            delete_model()
            get_model_state()
        except Exception as e:
            print 'error:', e.message
        time.sleep(5)
    pickle.dump(labeled_features, open('training_set.sav', 'wb'))
    
