# Title
  "Driver Drowsiness Detection"

## Team Members
  Jayda-Louise Nkansah(jnkansah), John Penola (Bjartskular8)

## Project Description
Dataset
This project will primarily use the Driver Drowsiness Dataset available on Kaggle, which contains labeled video frames and facial images capturing alert versus drowsy states across multiple subjects under varying lighting conditions. The dataset includes annotated eye states (open/closed), yawning events, and head pose angles collected from real driving scenarios. Supplementary data from the CEW (Closed Eyes in the Wild) dataset and the UTA Real-Life Drowsiness Dataset may be incorporated to improve generalization across diverse face types and environmental conditions.

Project Goal
The goal of this project is to build a real-time driver drowsiness detection system that continuously monitors a driver's face through a video feed and issues an alert before a fatigue-related incident occurs. Rather than treating this as a simple binary image classification task, the system will analyze temporal patterns across consecutive frames to distinguish between momentary eye closures and sustained drowsiness. The model will track three primary indicators: eye aspect ratio (EAR) to measure how open the eyes are over time, mouth aspect ratio (MAR) to detect yawning, and head pose estimation to identify nodding or tilting that signals loss of alertness.
The system will use a convolutional neural network for facial feature extraction combined with a sequence model to evaluate drowsiness across a rolling window of frames. When the model detects a sustained drowsy state above a defined threshold, it triggers a visual or audio alert. The final deliverable will be a working pipeline that accepts live webcam input or a pre-recorded video file, processes each frame in real time, and overlays drowsiness indicators directly onto the video output. Success will be measured by detection accuracy on held-out test sequences as well as the system's average response latency from onset of drowsiness to alert.
