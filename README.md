# Stereo Vision and Visual Odometry Project with KITTI Dataset

[![Version 1.0](https://img.shields.io/badge/Version-1.0-brightgreen)](https://github.com/your_username/your_project/releases/tag/v1.0)
[![Created on December 2024](https://img.shields.io/badge/Created%20on-December%202024-blue)](https://github.com/your_username/your_project/commit/your_commit_id)

## Group Members
- **Harris Nisar**: GitHub - [nisar2](https://github.com/nisar2)
- **Saharsh Sandeep Barve**: GitHub - [ssbarve2](https://github.com/Saharsh1005)

## Table of Contents
- [Project Description & Goals](#project-description--goals)
- [Member Roles](#member-roles)
- [Resources](#resources)
- [Results](#results)
- [Background](#background)
- [Acknowledgement](#acknowledgement)
- [References](#references)

## Project Description & Goals

### Description
This project focuses on understanding stereo vision based-visual odometry using the KITTI visual odometry dataset [3]. This dataset provides everything needed to build a visual odometry pipeline that can be experimented with, and that is exactly what was done. Specifically, this dataset consists of 22 stereo sequences, of which the first 11 also include the ground truth trajectory. The final 11 contain ground truths but are used by the authors of the dataset to evaluate performance of algorithms submitted by research groups. The dataset also includes camera calibration information. Using the stereo pairs, depth can be recovered using various algorithms. Here, depth is recovered using stock OpenCV functions and a deep learning approach using the ChiTransformer [4], with the goal to compare their performance. Then, for the next image,
features were matched between the left cameras of the current and subsequent frame. Features were represented using SIFT or ORB, again to compare their performance. To match features, brute force matching and KDTrees were explored. With the depth computed and the features matched, the motion of the camera was estimated. Specifically, the rotation and translation that best described how the matched 3D points changed between was computed using the Perspective-n-Point (PnP) algorithm. The rest of the report is organized as follows: first, the details of the dataset and the visual odometry pipeline are discussed; second, the trajectory computation results are shown for various configurations of the pipeline; finally, the approach is discussed, highlighting the strengths and weaknesses of the pipeline and suggestions for next steps to improve the results.

![Project Framework](/assets/flowchart_diagram.png)


### Goals
#### Preliminaries:
1. **Understanding the KITTI datasets:** Both team members will collaborate to familiarize themselves with the KITTI stereo and visual odometry datasets.
2. **Understanding basic mathematics and algorithms for stereo and visual odometry:** Both team members will acquire a strong theoretical foundation in 3D vision techniques.

#### Implement traditional stereo vision:
3. **Implement traditional stereo vision:** Both team members will work together to implement traditional stereo vision techniques using tutorials and resources.
4. **Test on KITTI stereo and visual odometry datasets:** The implemented traditional stereo vision will be tested on the KITTI datasets.
5. **Implement visual odometry:** Both team members will work on implementing visual odometry using tutorials.

#### Implement deep learning for stereo vision:
6. **Explore and run the deep learning model on the stereo dataset:** Harris will lead this effort, exploring and running deep learning models on the stereo dataset.
7. **Explore and run the deep learning model on the odometry dataset:** Saharsh will lead this part of the project, exploring and running deep learning models on the odometry dataset.
8. **Replace traditional stereo with deep learning in visual odometry and compare performance:** Both team members will collaborate to replace traditional stereo with deep learning in the visual odometry pipeline and evaluate their respective performances using labeled sequences from the KITTI dataset.

## Member Roles

- **Harris Nisar:**
    - Explore and run the deep learning model on the stereo dataset.
    
- **Saharsh Sandeep Barve:**
    - Explore and run the deep learning model on the odometry dataset.

Both team members will collaborate on other aspects of the project, including understanding the datasets, basic mathematics, implementing traditional stereo vision, implementing visual odometry, and comparing the impact of the two stereo vision approaches for visual odometry.

## Resources

### Data:
- **KITTI Stereo Dataset:** Contains 200 training scenes and 200 test scenes with four color images per scene, saved in lossless PNG format. Ground truth data is available for evaluation.
- **KITTI Visual Odometry Dataset:** Consists of 22 stereo sequences in lossless PNG format, with ground truth trajectories available for 11 training sequences (00-10).

### Implementation Platform:
- **Language:** Python
- **Libraries:** OpenCV, PIL, Numpy, PyTorch

### Resources and Tutorials:
- [Stereo Vision 1](https://github.com/savnani5/Depth-Estimation-using-Stereovision)
- [Visual Odometry 1](https://github.com/FoamoftheSea/KITTI_visual_odometry/tree/main)
- [Visual Odometry 2](https://jasleon.github.io/Visual-Odometry/)
- [Efficient Deep Learning for Stereo Matching](https://www.cs.toronto.edu/~urtasun/publications/luo_etal_cvpr16.pdf)
- Deep Learning for Depth Estimation:
    - [Repo 1](https://github.com/isl-cv/chitransformer)
    - [Repo 2](https://github.com/datvuthanh/Stereo-Matching)

### Computational Resources:
The project requires substantial computational resources, and the team has access to suitable hardware, including an RTX-3070 GPU, for both model training and inference.

## Project Results
### Project result video: 
[Link to Project Results Video](https://youtu.be/cJ4hgQHH6Ac)

### Raw Images
**Left Camera Image**
![Raw Image - Left Camera](/assets/raw_image_left.png)

**Right Camera Image**
![Raw Image - Right Camera](/assets/raw_image_right.png)

### Disparity Map SGBM from Classical Stereo Vision
![Disparity Map - Classical Stereo](/assets/classic_sgbm_disparity_map.png)

### Depth from Deep Learning Stereo Vision
![Deep Learning Stereo Depth Map](/assets/dl_stereo_depthmap.png)

### Odometry Results
#### Classical Stereo Vision
1. ![Odometry 00](/assets/classic_00_odometry.png)
2. ![Odometry 02](/assets/classic_02_odometry.png)
3. ![Odometry 05](/assets/classic_05_odometry.png)

#### Deep Learning Stereo Vision
1. ![Odometry 00](/assets/dl_00_odometry.png)
2. ![Odometry 02](/assets/dl_02_odometry.png)
3. ![Odometry 05](/assets/dl_05_odometry.png)

### Mean Squared Error (MSE) Error
![MSE Error](/assets/mse_error.png)

The above images showcase the results obtained from classical stereo vision, deep learning stereo vision, and the raw images used in the visual odometry pipeline. The odometry results for different sequences are presented for both classical and deep learning stereo vision. Additionally, the Mean Squared Error (MSE) error in depth estimation is visualized to evaluate the performance of the stereo vision methods.

For a more detailed analysis and discussion of the results, please refer to the corresponding sections in the report.

## Background

This project is significant for the team as it aligns with their work on a real-time medical instrument tracking system, which involves intelligent simulations and utilizes stereo vision techniques. Although the dataset and scale differ (cars vs. medical tools), the primary objective is to gain theoretical and practical expertise in 3D vision techniques, demystifying these complex algorithms. Additionally, implementing these algorithms in Python using familiar libraries like OpenCV and PyTorch will establish a solid foundation for their research work.

## Acknowledgement
We’d also like to acknowledge the work at [this](https://github.com/FoamoftheSea/KITTI_visual_odometry) repository for making the concepts of visual odometry easy to understand and providing a lot of code that we reused for the pipeline.

## References

[1] D. Scaramuzza and F. Fraundorfer, “Visual Odometry [Tutorial],” IEEE Robot. Autom. Mag., vol. 18, no. 4, pp. 80–92, Dec. 2011, doi: 10.1109/MRA.2011.943233.

[2] H. Moravec, “Obstacle Avoidance and Navigation in the Real World by a Seeing Robot Rover.” Sep. 1980. [Online]. Available: [Link to PDF](https://www.ri.cmu.edu/pub_files/pub4/moravec_hans_1980_1/moravec_hans_1980_1.pdf)

[3] A. Geiger, P. Lenz, and R. Urtasun, “Are we ready for autonomous driving? The KITTI vision benchmark suite,” in 2012 IEEE Conference on Computer Vision and Pattern Recognition, Providence, RI: IEEE, Jun. 2012, pp. 3354–3361. doi: 10.1109/CVPR.2012.6248074.

[4] Q. Su and S. Ji, “ChiTransformer:Towards Reliable Stereo from Cues,” 2022, doi: 10.48550/ARXIV.2203.04554.

[5] G. Bradski, “The OpenCV library,” Dr Dobbs J. Softw. Tools, 2000.

[6] D. G. Lowe, “Distinctive Image Features from Scale-Invariant Keypoints,” Int. J. Comput. Vis., vol. 60, no. 2, pp. 91–110, Nov. 2004, doi: 10.1023/B:VISI.0000029664.99615.94.

