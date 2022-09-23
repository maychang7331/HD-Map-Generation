import os
import numpy as np
import sys
import pcl
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # thesis
sys.path.append(BASE_DIR)

# Constants
data_dir = os.path.join(BASE_DIR, 'data')
labeled_shalun_dir = os.path.join(data_dir, 'labeled_shalun')

# Set Paths
pt_dir = os.path.join(labeled_shalun_dir, 'pt_9')
pc = pcl.load("other_9.las")
pc_tmp1 = pcl.PointCloud.make_voxel_grid_filter(pc)
pcl.VoxelGridFilter.set_leaf_size(pc_tmp1, 10.0, 10.0, 10.0)
pc_out = pcl.VoxelGridFilter.filter(pc_tmp1)

# ptCloudOut = pcl.pcdownsample(ptCloudIn, 'gridAverage', gridStep)