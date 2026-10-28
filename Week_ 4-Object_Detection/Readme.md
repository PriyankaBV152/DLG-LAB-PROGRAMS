# Week 04 - Object Detection Using Deep Learning

## Program Title

**Object Detection Using EfficientDet-D0 and VGG16 with Selective Search**

## Aim

To implement object detection using two pre-trained deep learning approaches and identify objects in images by drawing bounding boxes, displaying class labels, and showing confidence scores.

## Dataset Used

- **COCO Dataset (Common Objects in Context):** Used by the pre-trained EfficientDet-D0 model to detect objects belonging to 80 common object categories.
- **ImageNet Dataset:** Used to pre-train the VGG16 classification model, which recognizes 1,000 image categories.

**Input:** User-uploaded images in formats such as JPG and PNG.

No model training is performed. Both programs use pre-trained models.

## Program 4(a) - Object Detection Using EfficientDet-D0

### Description

EfficientDet-D0 is a pre-trained object detection model loaded using TensorFlow Hub. It detects objects in an image and returns their bounding-box coordinates, class IDs, and confidence scores.

### Algorithm

1. Import the required libraries.
2. Load the pre-trained EfficientDet-D0 model.
3. Upload an input image.
4. Convert the image into a TensorFlow tensor.
5. Perform object detection.
6. Filter detections using a confidence threshold of 0.5.
7. Draw bounding boxes and class labels on the image.
8. Display the final output image.

### Libraries Used

- TensorFlow
- TensorFlow Hub
- NumPy
- OpenCV

## Program 4(b) - Object Detection Using VGG16 and Selective Search

### Description

This program combines the VGG16 image classification model with Selective Search region proposals to demonstrate a simplified R-CNN-style object detection approach.

Selective Search generates candidate regions, VGG16 classifies the regions, and Non-Maximum Suppression (NMS) removes overlapping detections.

### Algorithm

1. Import the required libraries.
2. Load the pre-trained VGG16 model with ImageNet weights.
3. Upload and resize the input image.
4. Generate candidate regions using Selective Search.
5. Resize each region to 224 × 224 pixels.
6. Classify the regions using VGG16.
7. Retain predictions with confidence scores above 0.90.
8. Apply Non-Maximum Suppression to reduce duplicate bounding boxes.
9. Draw bounding boxes and labels on the image.
10. Display the final output image.

### Libraries Used

- TensorFlow and Keras
- NumPy
- OpenCV with the contrib module

**Note:** VGG16 is an image classifier rather than a dedicated object detector. This implementation is an educational approximation of an R-CNN-style pipeline.

## Result

Both programs demonstrate object detection on user-provided images using pre-trained deep learning models.

- **Program 1:** EfficientDet-D0 identifies objects and displays bounding boxes, class labels, and confidence scores.
- **Program 2:** VGG16 combined with Selective Search classifies candidate image regions, filters predictions, and displays bounding boxes after Non-Maximum Suppression.

The results depend on the input image, model predictions, and confidence thresholds.

## Files

- Program4a(EfficientDet-D0).ipynb - Object detection using the pre-trained EfficientDet-D0 model.
- Program4b(VGG16-Selective-Search).ipynb - Object detection using VGG16, Selective Search, and Non-Maximum Suppression (NMS).