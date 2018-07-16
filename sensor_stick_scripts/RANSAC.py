# Import PCL module
import pcl

# Load Point Cloud file
cloud = pcl.load_XYZRGB('tabletop.pcd')


# Voxel Grid filter
# Create a VoxelGrid filter object for our input point cloud
vox = cloud.make_voxel_grid_filter()

# Choose a voxel (also known as leaf) size
# Note: this (1) is a poor choice of leaf size   
# Experiment and find the appropriate size!
LEAF_SIZE = .01

# Set the voxel (or leaf) size  
vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)

# Call the filter function to obtain the resultant downsampled point cloud
cloud_filtered = vox.filter()
pcl.save(cloud_filtered, 'voxel_downsampled.pcd')

# PassThrough filter
passthrough = cloud_filtered.make_passthrough_filter()
passthrough.set_filter_field_name('z')
passthrough.set_filter_limits(0.6, 1.8)
cloud_filtered = passthrough.filter()
passthrough = cloud_filtered.make_passthrough_filter()
passthrough.set_filter_field_name('y')
passthrough.set_filter_limits(-3, 3)
cloud_filtered = passthrough.filter()
pcl.save(cloud_filtered, 'pass_through_filtered.pcd')

# RANSAC plane segmentation
# Create the segmentation object
seg = cloud_filtered.make_segmenter()

# Set the model you wish to fit 
seg.set_model_type(pcl.SACMODEL_PLANE)
seg.set_method_type(pcl.SAC_RANSAC)

# Max distance for a point to be considered fitting the model
# Experiment with different values for max_distance 
# for segmenting the table
max_distance = .01
seg.set_distance_threshold(max_distance)

# Call the segment function to obtain set of inlier indices and model coefficients
inliers, coefficients = seg.segment()

# Extract inliers
extracted_inliers = cloud_filtered.extract(inliers, negative=False)
extracted_outliers = cloud_filtered.extract(inliers, negative=True)
pcl.save(extracted_inliers, 'extracted_inliers.pcd')
pcl.save(extracted_outliers, 'extracted_outliers.pcd')

# Extract outliers
# Much like the previous filters, we start by creating a filter object:
outlier_filter = cloud_filtered.make_statistical_outlier_filter()
outlier_filter.set_mean_k(50)
x = 1
outlier_filter.set_std_dev_mul_thresh(x)
cloud_filtered = outlier_filter.filter()
pcl.save(cloud_filtered, 'outlier_filtered.pcd')
