[![Udacity - Robotics NanoDegree Program](https://s3-us-west-1.amazonaws.com/udacity-robotics/Extra+Images/RoboND_flag.png)](https://www.udacity.com/robotics)

## 3D Perception Project
### Writeup by Ayo Adedeji
---
#### Instructions on how to setup Gazebo, ROS and other dependencies can be found [here](./project_setup.md).

## [Rubric](https://review.udacity.com/#!/rubrics/1067/view) Points
---
### Perception Pipeline Implementation of [RoboND Exercises 1, 2 and 3](https://github.com/udacity/RoboND-Perception-Exercises)
#### 1. Complete Exercise 1 steps | Pipeline for statistical outlier, voxel grid, passthrough filtering and RANSAC plane fitting implemented:

Below images are based on [test1.world](https://github.com/theayoad/RoboND-Perception-Project/blob/master/pr2_robot/worlds/test1.world)

##### Statistical Outlier Removal (k = 50, x = .05) 
</br>
* The pcl.StatisticalOutlierRemovalFilter computes the mean distance of each point to k number of neighbors. All points whose mean distance (to neighbors) is greater than the global_mean_distance + x * global_std_dev are considered to be outliers and removed from the point cloud.

<p align="center"> <img src="./output/world1_statistical_outlier_filter_output.png"> </p>

```python
# Statistical Outlier Filtering
outlier_filter = point_cloud.make_statistical_outlier_filter()
outlier_filter.set_mean_k(50) # num of neighboring points to analyze
x = .05 # threshold scale factor
outlier_filter.set_std_dev_mul_thresh(x) # outlier > global_mean_distance + x*global_std_dev
point_cloud_filtered = outlier_filter.filter()
```

##### Voxel Grid Downsampling (leaf_size = .005)
</br>
* The pcl.VoxelGridFilter class assembles a local 3D grid over a given PointCloud and downsamples the point cloud data based on user specified voxel grid leaf size.

<p align="center"> <img src="./output/world1_voxel_grid_downsample_output.png"> </p>

```python
# Voxel Grid Downsampling
vox = point_cloud_filtered.make_voxel_grid_filter()
leaf_size = .005 # voxel (leaf) size
vox.set_leaf_size(leaf_size, leaf_size, leaf_size)
point_cloud_filtered = vox.filter()
```

##### PassThrough  (y=[-.42, .42], z=[.6, 1.8])
</br>
* The pcl.PassThroughFilter class removes points from point cloud that do not meet constraints / limits for a particular field of the point type.
* Z-axis filtering in range of [.6, 1.8] ensures only table + objects data make up point cloud
* Y-axis filtering in range of [-.42, .42] prevents bin recognition

<p align="center"> <img src="./output/world1_passthrough_filter_output.png"> </p>

```python
# PassThrough Filter
passthrough = point_cloud_filtered.make_passthrough_filter()
passthrough.set_filter_field_name('y')
passthrough.set_filter_limits(-0.42, 0.42) # prevent bin recognition
point_cloud_filtered = passthrough.filter()
passthrough = point_cloud_filtered.make_passthrough_filter()
passthrough.set_filter_field_name('z')
passthrough.set_filter_limits(0.6, 1.8) # only table + objects
point_cloud_filtered = passthrough.filter()
```

##### RANSAC Plane Fitting (dist=.01) 
</br>
* The pcl.Segmentation class runs sample consensus methods and models. Specifically, the RANSAC algorithm involves iterated hypothesis and verification of point cloud data: a hypothetical shape of the specified model (pcl.SACMODEL_PLANE) is generated  by selecting a minimal subset of n-points at random and evaluating the corresponding shape to model fit. 

<p align="center"> <img src="./output/world1_cloud_table.png"> </p>
<p align="center"> <img src="./output/world1_cloud_objects.png"> </p>

```python
# RANSAC Plane Segmentation
seg = point_cloud_filtered.make_segmenter()
seg.set_model_type(pcl.SACMODEL_PLANE)
seg.set_method_type(pcl.SAC_RANSAC)
seg.set_distance_threshold(.01)
inliers, coefficients = seg.segment()
# Extract inliers and outliers
cloud_table = point_cloud_filtered.extract(inliers, negative=False)
cloud_objects = point_cloud_filtered.extract(inliers, negative=True)
```


#### 2. Complete Exercise 2 steps: Pipeline including clustering for segmentation implemented.  

#### 2. Complete Exercise 3 Steps.  Features extracted and SVM trained.  Object recognition implemented.
Here is an example of how to include an image in your writeup.

![demo-1](https://user-images.githubusercontent.com/20687560/28748231-46b5b912-7467-11e7-8778-3095172b7b19.png)

### Pick and Place Setup

#### 1. For all three tabletop setups (`test*.world`), perform object recognition, then read in respective pick list (`pick_list_*.yaml`). Next construct the messages that would comprise a valid `PickPlace` request output them to `.yaml` format.

And here's another image! 
![demo-2](https://user-images.githubusercontent.com/20687560/28748286-9f65680e-7468-11e7-83dc-f1a32380b89c.png)

Spend some time at the end to discuss your code, what techniques you used, what worked and why, where the implementation might fail and how you might improve it if you were going to pursue this project further.  



