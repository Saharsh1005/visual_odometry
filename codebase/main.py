
'''
Reference: https://github.com/FoamoftheSea/KITTI_visual_odometry
'''

import cv2
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from classicalStereo.datasetHandler import Dataset_Handler
from classicalStereo.utils import compute_left_disparity_map, decompose_projection_matrix, stereo_2_depth, pointcloud2image, \
                                  match_features, extract_features, estimate_motion, filter_matches_distance, calc_depth_map
from dlStereo.dataHandler_ChiTransformer import ChiTransformer_StereoDepth

from utils import plot_odometry

def visual_odometry_deepLearning(handler, detector='sift', matching='BF', filter_match_distance=None,
                    stereo_matcher='sgbm', mask=None, subset=None, plot=False, dl_handler= ChiTransformer_StereoDepth(), run_for=None):

    
    # Report methods being used to user
    print('Generating disparities with Stereo{}'.format(str.upper(stereo_matcher)))
    print('Detecting features with {} and matching with {}'.format(str.upper(detector),
                                                                  matching))
    if filter_match_distance is not None:
        print('Filtering feature matches at threshold of {}*distance'.format(filter_match_distance))

    if subset is not None:
        num_frames = subset
    else:
        num_frames = handler.num_frames
        
    if plot:
        fig = plt.figure(figsize=(14, 14))
        ax = fig.add_subplot(projection='3d')
        ax.view_init(elev=-20, azim=270)
        xs = handler.gt[:, 0, 3]
        ys = handler.gt[:, 1, 3]
        zs = handler.gt[:, 2, 3]
        ax.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs)))
        ax.plot(xs, ys, zs, c='k')
        
    # Establish a homogeneous transformation matrix. First pose is identity
    T_tot = np.eye(4)
    trajectory = np.zeros((num_frames, 3, 4))
    trajectory[0] = T_tot[:3, :]
    imheight = handler.imheight
    imwidth = handler.imwidth
    
    # Decompose left camera projection matrix to get intrinsic k matrix
    k_left, r_left, t_left = decompose_projection_matrix(handler.P0)
    
    if handler.low_memory:
        handler.reset_frames()
        image_plus1 = next(handler.images_left)
        
    # Iterate through all frames of the sequence
    for i in range(num_frames-1):
        start = datetime.datetime.now()
        
        if handler.low_memory:
            image_left = image_plus1
            image_plus1 = next(handler.images_left)
            image_right = next(handler.images_right)
        else:
            image_left = handler.images_left[i]
            image_plus1 = handler.images_left[i+1]
            image_right = handler.images_right[i]
        print('Depth frame: ',i)
        depth = dl_handler.get_depthMap(i)
        depth = 1/depth

        # Get keypoints and descriptors for left camera image of two sequential frames
        kp0, des0 = extract_features(image_left, detector, mask)
        kp1, des1 = extract_features(image_plus1, detector, mask)
        
        # Get matches between features detected in two subsequent frames
        matches_unfilt = match_features(des0, 
                                        des1,
                                        matching=matching,
                                        detector=detector
                                       )
        # print('Number of features before filtering: ', len(matches_unfilt))
        
        # Filter matches if a distance threshold is provided by user
        if filter_match_distance is not None:
            matches = filter_matches_distance(matches_unfilt, filter_match_distance)
        else:
            matches = matches_unfilt
         
        # Estimate motion between sequential images of the left camera
        rmat, tvec, img1_points, img2_points = estimate_motion(matches,
                                                               kp0,
                                                               kp1,
                                                               k_left,
                                                               depth
                                                              )
        
        # Create a blank homogeneous transformation matrix
        Tmat = np.eye(4)
        Tmat[:3, :3] = rmat
        Tmat[:3, 3] = tvec.T
        
        T_tot = T_tot.dot(np.linalg.inv(Tmat))
        
        trajectory[i+1, :, :] = T_tot[:3, :]
        
        end = datetime.datetime.now()
        print('Time to compute frame {}:'.format(i+1), end-start)
        
        
    return trajectory

def visual_odometry(handler, detector='sift', matching='BF', filter_match_distance=None,
                    stereo_matcher='sgbm', mask=None, subset=None, plot=False):
    # Determine if handler has lidar data
    lidar = False #handler.lidar
    
    # Report methods being used to user
    print('Generating disparities with Stereo{}'.format(str.upper(stereo_matcher)))
    print('Detecting features with {} and matching with {}'.format(str.upper(detector),
                                                                  matching))
    if filter_match_distance is not None:
        print('Filtering feature matches at threshold of {}*distance'.format(filter_match_distance))
    if lidar:
        print('Improving stereo depth estimation with lidar data')
    if subset is not None:
        num_frames = subset
    else:
        num_frames = handler.num_frames
        
    if plot:
        fig = plt.figure(figsize=(14, 14))
        ax = fig.add_subplot(projection='3d')
        ax.view_init(elev=-20, azim=270)
        xs = handler.gt[:, 0, 3]
        ys = handler.gt[:, 1, 3]
        zs = handler.gt[:, 2, 3]
        ax.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs)))
        ax.plot(xs, ys, zs, c='k')
        
    # Establish a homogeneous transformation matrix. First pose is identity
    T_tot = np.eye(4)
    trajectory = np.zeros((num_frames, 3, 4))
    trajectory[0] = T_tot[:3, :]
    imheight = handler.imheight
    imwidth = handler.imwidth
    
    # Decompose left camera projection matrix to get intrinsic k matrix
    k_left, r_left, t_left = decompose_projection_matrix(handler.P0)
    
    if handler.low_memory:
        handler.reset_frames()
        image_plus1 = next(handler.images_left)
        
    # Iterate through all frames of the sequence
    for i in range(num_frames - 1):
        start = datetime.datetime.now()
        
        if handler.low_memory:
            image_left = image_plus1
            image_plus1 = next(handler.images_left)
            image_right = next(handler.images_right)
        else:
            image_left = handler.images_left[i]
            image_plus1 = handler.images_left[i+1]
            image_right = handler.images_right[i]
            
        depth = stereo_2_depth(image_left,
                               image_right,
                               P0=handler.P0,
                               P1=handler.P1,
                               matcher=stereo_matcher
                              )

        if lidar:
            if handler.low_memory:
                pointcloud = next(handler.pointclouds)
            else:
                pointcloud = handler.pointclouds[i]
            
            lidar_depth = pointcloud2image(pointcloud,
                                           imheight=imheight,
                                           imwidth=imwidth,
                                           Tr=handler.Tr,
                                           P0=handler.P0
                                          )
            indices = np.where(lidar_depth > 0)
            depth[indices] = lidar_depth[indices]
            
        # Get keypoints and descriptors for left camera image of two sequential frames
        kp0, des0 = extract_features(image_left, detector, mask)
        kp1, des1 = extract_features(image_plus1, detector, mask)
        
        # Get matches between features detected in two subsequent frames
        matches_unfilt = match_features(des0, 
                                        des1,
                                        matching=matching,
                                        detector=detector
                                       )
        
        # Filter matches if a distance threshold is provided by user
        if filter_match_distance is not None:
            matches = filter_matches_distance(matches_unfilt, filter_match_distance)
        else:
            matches = matches_unfilt

        # Estimate motion between sequential images of the left camera
        rmat, tvec, img1_points, img2_points = estimate_motion(matches,
                                                               kp0,
                                                               kp1,
                                                               k_left,
                                                               depth
                                                              )
        
        # Create a blank homogeneous transformation matrix
        Tmat = np.eye(4)
        Tmat[:3, :3] = rmat
        Tmat[:3, 3] = tvec.T
        
        T_tot = T_tot.dot(np.linalg.inv(Tmat))
        
        trajectory[i+1, :, :] = T_tot[:3, :]
        
        end = datetime.datetime.now()
        print('Time to compute frame {}:'.format(i+1), end-start)
        
    return trajectory


if __name__ == "__main__":
    
    input_path              = 'C:/Users/nisar2/Desktop/final_543/database'
    dl_stereo_depth_path    = 'C:/Users/nisar2/Desktop/final_543/inferences'
    
    sequences               = ['00','02','05']
    output_path             = 'C:/Users/nisar2/Desktop/final_543/infer_temp' # Store odometry results

    flag_dl_odometry        = True
    flag_classical_odometry = True

    print('Running Visual Odometry main.py\n')

    if not (flag_dl_odometry or flag_classical_odometry):
        raise AssertionError("Select at least one odometry method (DL or classic)")

    for i in range(len(sequences)):

        dl_stereo_path = f'{dl_stereo_depth_path}/{sequences[i]}'
        handler    = Dataset_Handler(sequence = sequences[i], db_fp = input_path)                         # read images
        dl_handler = ChiTransformer_StereoDepth(db_fp = dl_stereo_path)                                   # read DL stereo depth maps
        
        if flag_classical_odometry:
            print('\nRunning ClassicalStereo Odometry...')
            classic_trajectory = visual_odometry(handler, 
                                        detector='sift',
                                        matching='BF',
                                        filter_match_distance=0.3,
                                        stereo_matcher='sgbm',
                                        mask= None, # mask
                                        subset=None,
                                        plot=False,
                                        )
            
            np.save(f'{output_path}/trajectory-{sequences[i]}_classic.npy', classic_trajectory)

        if flag_dl_odometry:
            print('\nRunning DLStereo Odometry...')
            dl_trajectory = visual_odometry_deepLearning(handler, 
                                        detector='sift',
                                        matching='BF',
                                        filter_match_distance=0.3,
                                        stereo_matcher='sgbm',
                                        mask= None,
                                        subset=None,
                                        plot=False,
                                        dl_handler = dl_handler
                                        
                                        )
            np.save(f'{output_path}/trajectory-{sequences[i]}_dl.npy', dl_trajectory)




