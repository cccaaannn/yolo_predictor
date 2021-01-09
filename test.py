import os
from yolo_predictor import yolo_predictor

if __name__ == "__main__":
    model_path = "model_files/yolov4_coco.weights"
    names_path = "model_files/coco.names"
    test_images_path = "test_images"

    yp = yolo_predictor(model_path, names_path)

    for image_path in sorted(os.listdir(test_images_path)):
        detection = yp.detect(os.path.join(test_images_path, image_path))
        print(detection)