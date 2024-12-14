#YOLOv3 Real-Time Object Detection
Description
This project implements a real-time object detection system using YOLOv3 (You Only Look Once), OpenCV, and the COCO dataset. The program captures live video streams from a webcam, processes each frame to detect objects, and displays the detected objects with bounding boxes and class labels.

Features
Real-time object detection and classification using YOLOv3.
Live video input from the webcam.
Integration with the COCO dataset for object recognition.
Non-Max Suppression for improved accuracy and fewer duplicate boxes.
Requirements
1.Python 3.x
2.OpenCV
3.NumPy
4.YOLOv3 weights, configuration, and COCO class names
Setup
Clone the repository

bash
Copy code
git clone https://github.com/mohammedabsal/Object-Detection.git
cd yolo-real-time-detection
Download YOLOv3 files:

Download yolov3.weights from YOLO official weights.
Download yolov3.cfg and coco.names files.
Install Dependencies

bash
Copy code
pip install opencv-python numpy
Run the Program

bash
Copy code
python yolo_object_detection.py
Press 'q' to exit the live video feed.

Code Explanation
YOLOv3 Integration:

cv2.dnn.blobFromImage preprocesses video frames for YOLO input.
Forward pass is run through the YOLO network to get detection predictions.
Bounding Boxes:

Detected objects are marked with bounding boxes and labeled using the COCO dataset classes.
Non-Max Suppression:

Removes redundant overlapping boxes to improve detection accuracy.
