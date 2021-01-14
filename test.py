import cv2
import os

from yolo_predictor import yolo_predictor
from yolo_drawer import yolo_drawer

if __name__ == "__main__":
    model_path = "model_files/yolov4_coco.weights"
    names_path = "model_files/coco.names"
    test_images_path = "test_images"

    predictor = yolo_predictor(model_path, names_path)
    drawer = yolo_drawer(random_seed_start_point = 10)  # seed for changing colors consistently

    for image_path in sorted(os.listdir(test_images_path)):
        image_full_path = os.path.join(test_images_path, image_path)

        try:
            predictions = predictor.predict(image_full_path)
            image, save_path = drawer.draw(predictions, image_full_path, show=True, resize=False, save_folder_path="test_results", saved_file_suffix="_predicted")
            print(predictions, save_path)
        except:
            print("file not supported")
