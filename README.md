# Monocular Visual Odometry Pipeline

A feature-based monocular visual odometry system implemented in Python/OpenCV for real-time camera trajectory estimation and sparse 3D reconstruction.

![Trajectory Preview](assets/teaser.png)

---

## Overview

This project implements a complete visual odometry pipeline capable of:

- Feature detection and matching
- Two-view geometry estimation
- Triangulation of 3D landmarks
- Camera pose estimation
- Landmark tracking across frames
- Sparse map generation

The system estimates camera motion from monocular image sequences while reconstructing a sparse representation of the environment.

---

## Demo Results

### Proj 1 Reconstruction

![Problem 1](results/prob1_map.png)

### Proj 2 Reconstruction

![Problem 2](results/prob2_map.png)

### Bootstrap Initialization

![Bootstrap](results/prob2_bootstrap.png)

---

## Pipeline

```text
Input Frames
     ↓
Feature Detection
     ↓
Feature Matching
     ↓
Essential Matrix Estimation
     ↓
Pose Recovery
     ↓
Triangulation
     ↓
Landmark Tracking
     ↓
Continuous Pose Estimation
```

---

## Repository Structure

```text
visual-odometry/
├── notebooks/        # Experiment notebooks
├── src/              # Core implementation
├── results/          # Generated maps and visualizations
├── assets/           # README assets
└── data/             # Dataset instructions
```

---

## Methods Used

### Feature Detection
- Shi-Tomasi Corner Detection
- ORB/SIFT feature descriptors

### Geometry
- Essential Matrix estimation using RANSAC
- Epipolar geometry constraints
- Perspective-n-Point (PnP)

### Mapping
- Linear triangulation
- Landmark reprojection filtering
- Keypoint tracking

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/visual-odometry.git
cd visual-odometry
```

Create environment:

```bash
pip install -r requirements.txt
```

---

## Usage

Run the notebook experiments:

```bash
jupyter notebook notebooks/project_1.ipynb
```

or

```bash
jupyter notebook notebooks/project_2.ipynb
```

---

## Results

| Sequence | Status | Notes |
|---|---|---|
| Problem 1 | Complete | Stable trajectory estimation |
| Problem 2 | Complete | Improved bootstrap initialization |

---

## Future Improvements

- Bundle Adjustment
- Loop Closure
- Real-time optimization
- Dense reconstruction
- ROS integration

---

## Technologies

- Python
- OpenCV
- NumPy
- Matplotlib
- Jupyter Notebook

---

## Author

Isabella Opoku-Ware

Graduate Student in Electrical Engineering — Machine Learning  
George Washington University

---

## References

- Multiple View Geometry — Hartley & Zisserman
- OpenCV Visual Odometry Documentation
- Feature-based SLAM literature