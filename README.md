# YOLOv3 Real-Time Object Detection

## Description  
This project implements a **real-time object detection system** using **YOLOv3** (You Only Look Once), **OpenCV**, and the **COCO dataset**. The program captures live video streams from a webcam, processes each frame to detect objects, and displays the detected objects with bounding boxes and class labels.

## Features  
- Real-time object detection and classification using YOLOv3.  
- Live video input from the webcam.  
- Integration with the COCO dataset for object recognition.  
- Non-Max Suppression for improved accuracy and fewer duplicate boxes.  

## Requirements  
- Python 3.x  
- OpenCV  
- NumPy  
- YOLOv3 weights, configuration, and COCO class names  

## Setup  

1. **Clone the repository**  
   ```bash
   git clone https://github.com/yourusername/Object-Detection.git
   cd Object-Detection
   ```

2. **Download YOLOv3 files**:  
   - Download `yolov3.weights` from [YOLO official weights](https://pjreddie.com/media/files/yolov3.weights).  
   - Download `yolov3.cfg` and `coco.names` files.  

3. **Install Dependencies**  
   ```bash
   pip install opencv-python numpy
   ```

4. **Run the Program**  
   ```bash
   python yolo_object_detection.py
   ```

5. **Press 'q'** to exit the live video feed.  

## Code Explanation  
- **YOLOv3 Integration**:  
   - `cv2.dnn.blobFromImage` preprocesses video frames for YOLO input.  
   - Forward pass is run through the YOLO network to get detection predictions.  

- **Bounding Boxes**:  
   - Detected objects are marked with bounding boxes and labeled using the COCO dataset classes.  

- **Non-Max Suppression**:  
   - Removes redundant overlapping boxes to improve detection accuracy.  
