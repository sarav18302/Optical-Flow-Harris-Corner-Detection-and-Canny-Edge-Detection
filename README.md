# Optical Flow, Harris Corner Detection, and Canny Edge Detection Project Documentation

This repository implements computer vision algorithms, including Optical Flow, Harris Corner Detection, and Canny Edge Detection. It uses Jupyter Notebooks for experimentation and analysis, targeting image processing and feature detection tasks. The project demonstrates the application of fundamental vision techniques for motion estimation and edge/corner identification.

## Overview

This project focuses on implementing and demonstrating three key computer vision algorithms: Optical Flow, Harris Corner Detection, and Canny Edge Detection. These algorithms are widely used for motion tracking, feature detection, and edge detection in images. The repository is designed for research, experimentation, and educational purposes, providing modular implementations in Jupyter Notebooks. 

Goals include: 
- Showcasing algorithmic principles through clear Python implementations.
- Enabling reproducibility and experimentation via Jupyter Notebooks.
- Facilitating modifications for domain-specific adaptations.
Rationale: dividing the project into independent modules ensures modularity, allowing developers to focus on specific algorithms or integrate them into larger pipelines. The repository adopts a procedural approach, prioritizing clarity and reusability.



## Key Features

This project provides the following features:
- Optical Flow computation using the Lucas-Kanade method for motion estimation.
- Harris Corner Detection for identifying corner features in images.
- Canny Edge Detection for detecting edges with adaptive thresholds.
- Visualization of results for analysis and debugging.
- Modular and reusable implementation for easy integration into larger systems.
- Parameter customization for fine-tuning algorithm behavior.
Additional capabilities include:
- Support for grayscale image processing.
- Visualization overlays for feature points (corners, edges, or motion vectors).

## Canny Edge Detector – Implementation Details

The Canny Edge Detector in this repository is implemented **from scratch**, following the classical multi-stage pipeline. Each stage is modularized into well-defined functions to improve clarity, experimentation, and benchmarking against OpenCV’s implementation.

### Implemented Components

#### 1. Noise Modeling
- **Impulse (Salt-and-Pepper) Noise Injection**
  - Function: `add_impulse_noise(image, noise_prob)`
  - Used to evaluate edge detector robustness under sparse high-intensity noise.
- **Gaussian Noise Injection**
  - Function: `add_gaussian_noise(image, mean, std)`
  - Simulates sensor noise and smooth intensity perturbations.

#### 2. Image Denoising / Smoothing
- **Box Filter (Mean Filter)**
  - Function: `apply_box_filter(image, kernel_size)`
  - Baseline linear smoothing for noise reduction.
- **Gaussian Filter**
  - Kernel generation: `create_kernel(kernel_size, sigma)`
  - Convolution: `apply_gaussain_filter(image, kernel_size, sigma)`
  - Primary smoothing step before gradient computation.
- **Median Filter**
  - Function: `apply_median_filter(image, kernel_size)`
  - Effective against impulse noise while preserving edges.

#### 3. Gradient Computation
- **Sobel Operator**
  - Function: `sobel_filters(img)`
  - Computes horizontal and vertical gradients, gradient magnitude, and gradient orientation.
- **Gradient Normalization**
  - Function: `scale(x)`
  - Normalizes gradient magnitudes for visualization and thresholding.

#### 4. Non-Maximum Suppression
- Function: `non_max_suppression(G, theta)`
- Suppresses non-maximum gradient responses along the gradient direction.
- Produces one-pixel-wide edge candidates.

#### 5. Double Thresholding
- Function: `threshold(image, lowthreshold, highthreshold)`
- Classifies pixels into:
  - Strong edges
  - Weak edges
  - Non-edges

#### 6. Edge Tracking by Hysteresis
- Function: `hysteresis(image)`
- Retains weak edges only if connected to strong edges.
- Eliminates isolated responses caused by noise.

#### 7. End-to-End Custom Canny Pipeline
- Function: `apply_canny_edge_detector_self(image, threshold1, threshold2)`
- Integrates Gaussian smoothing, Sobel gradients, non-maximum suppression, double thresholding, and hysteresis.

#### 8. OpenCV Canny Benchmarking
- Function: `apply_canny_edge_detector_opencv(image, threshold1, threshold2)`
- Enables direct comparison with OpenCV’s built-in Canny implementation.

#### 9. Quantitative Evaluation
- **RMSE Error Metric**
  - Function: `rmse(image1, image2)`
  - Computes root mean square error between custom and OpenCV outputs.

#### 10. Visualization Utilities
- Function: `show_image_grid(images, M, N, title)`
- Supports side-by-side visualization of noise effects, filtering stages, and final edge maps.

## Harris Corner Detection – Implementation Details

The Harris Corner Detector in this repository is implemented **from scratch**, closely following the classical Harris-Stephens formulation. The implementation emphasizes mathematical clarity, step-by-step processing, and modular design to enable experimentation and comparison with OpenCV’s built-in detector.

### Implemented Components

#### 1. Image Preprocessing
- **Grayscale Conversion**
  - Ensures consistent intensity representation before gradient computation.
- **Gaussian Smoothing**
  - Reduces noise prior to derivative calculation.
  - Prevents spurious corner responses caused by high-frequency noise.

#### 2. Image Gradient Computation
- **Sobel Derivative Filters**
  - Computes first-order image gradients:
    - Horizontal gradient (`Ix`)
    - Vertical gradient (`Iy`)
  - Forms the basis for local structure analysis.

#### 3. Structure Tensor (Second-Moment Matrix)
- Computes the elements of the second-moment matrix:
  - `Ix²`, `Iy²`, and `IxIy`
- Applies Gaussian smoothing to these terms to aggregate gradient information over a local neighborhood.

#### 4. Harris Corner Response Calculation
- Implements the Harris response function:
  - `R = det(M) − k * (trace(M))²`
- Parameter `k` controls sensitivity to corner versus edge responses.
- Generates a corner strength map over the image.

#### 5. Thresholding
- Applies thresholding on the Harris response map.
- Suppresses weak responses to retain only strong corner candidates.

#### 6. Non-Maximum Suppression
- Retains only local maxima in the Harris response map.
- Eliminates clustered or redundant corner detections.
- Produces well-localized corner points.

#### 7. Corner Visualization
- Overlays detected corners on the original image.
- Enables qualitative inspection of:
  - Corner localization accuracy
  - Robustness across textured and flat regions

#### 8. Parameter Tuning and Analysis
- Supports experimentation with:
  - Gaussian kernel size
  - Harris constant `k`
  - Threshold values
- Facilitates understanding of detector behavior under different settings.

#### 9. OpenCV Harris Comparison
- Uses OpenCV’s Harris Corner Detector as a baseline.
- Enables side-by-side comparison between:
  - Custom implementation
  - Library-based implementation

#### 10. Visualization Utilities
- Displays intermediate results including:
  - Gradient maps
  - Harris response heatmaps
  - Final corner detections
- Aids debugging and algorithm interpretation.

## Optical Flow – Implementation Details

The Optical Flow component in this repository is implemented **from scratch**, focusing on the classical **Lucas–Kanade method** for motion estimation and its application to **image warping**. The implementation emphasizes pixel-wise motion estimation, numerical stability, and visualization on real video sequences.

### Implemented Components

#### 1. Video Frame Handling
- Loads video sequences as ordered frame arrays.
- Supports multiple videos with consistent frame dimensions.
- Converts frames to grayscale for optical flow computation.

#### 2. Image Visualization Utilities
- **Grid Visualization**
  - Function: `show_image_grid(images, M, N, title)`
  - Displays multiple frames or results in a structured grid.
- **GIF Generation**
  - Function: `images_to_gif(frames, video_id)`
  - Converts processed frame sequences into animated GIFs for qualitative evaluation.

#### 3. Lucas–Kanade Optical Flow (From Scratch)
- Function: `find_optical_flow(old_frame, new_frame, window_size, min_quality)`
- Key steps:
  - Intensity normalization
  - Spatial gradient computation using custom convolution kernels (`Ix`, `Iy`)
  - Temporal gradient computation (`It`)
  - Local window-based least squares solution using the pseudo-inverse
- Estimates per-pixel motion vectors `(u, v)`.

#### 4. Motion Vector Visualization
- Draws **arrowed motion vectors** directly on RGB frames.
- Applies thresholding on large displacements to suppress unstable vectors.
- Enables intuitive visualization of motion direction and magnitude.

#### 5. Sequential Optical Flow on Videos
- Function: `optical_flow(frames, video_id)`
- Computes optical flow between consecutive frames.
- Produces animated GIFs showing motion evolution across time.

#### 6. Dense Optical Flow for Warping
- Extended optical flow computation between distant frames `[i, i+8]`.
- Iterative refinement using:
  - Sobel spatial gradients
  - Gaussian smoothing of flow fields
- Produces dense flow maps `(flow_x, flow_y)`.

#### 7. Image Warping Using Optical Flow
- Function: `optical_flow_warping(frame_a, frame_b)`
- Uses computed optical flow to warp frame `i` toward frame `i+8`.
- Implements **forward warping** from scratch.

#### 8. Interpolation Methods
- **Bilinear Interpolation**
  - Function: `interpolate_bilinear(grid, query_points)`
  - Used during warping for subpixel accuracy.
- **Bicubic-like Hole Filling**
  - Function: `bicubic_interpolate(img)`
  - Fills missing pixels after warping using weighted neighborhood interpolation.

#### 9. Warped Frame Evaluation
- Compares:
  - Warped frame generated via optical flow
  - Actual target frame `(i+8)`
- Enables qualitative assessment of motion estimation accuracy.

#### 10. End-to-End Optical Flow Pipeline
- Integrates:
  - Frame preprocessing
  - Optical flow estimation
  - Motion visualization
  - Image warping and interpolation



## Technology Stack

The project leverages the following technologies:
- Python: Primary language for implementation.
- Jupyter Notebook: Interactive environment for experimentation and step-by-step execution.
- OpenCV: Core library for image processing operations and visualization.
- NumPy: Efficient numerical computations and array manipulations.
- Matplotlib: Visualization of image processing results.
