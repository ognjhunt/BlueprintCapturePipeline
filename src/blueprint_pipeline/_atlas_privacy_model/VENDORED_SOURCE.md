`face_detection_yunet_2023mar.onnx` and `LICENSE` are the unmodified YuNet face
detector and its MIT license from the OpenCV Model Zoo
(`opencv/opencv_zoo`, `models/face_detection_yunet/`, fetched 2026-10-22).

- sha256: `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`
- size: 232589 bytes
- runtime: `cv2.FaceDetectorYN`, present in both the locked OpenCV 4.11 and the
  OpenCV 5.0 that production currently runs (OpenCV 5 moved the Haar cascades to
  contrib, so they are not an option).

`atlas_privacy_screen` refuses to run when these bytes do not match the digest.
Keep the license with the model on every redistribution.
