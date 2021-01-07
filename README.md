## Yolo predictor
### Predict images with tensorflow converted darknet yolov4 model in a single line.
---

![GitHub top language](https://img.shields.io/github/languages/top/cccaaannn/yolo_predictor?style=flat-square) ![](https://img.shields.io/github/repo-size/cccaaannn/yolo_predictor?style=flat-square) [![GitHub license](https://img.shields.io/github/license/cccaaannn/yolo_predictor?style=flat-square)](https://github.com/cccaaannn/yolo_predictor/blob/master/LICENSE)


## Before starting
- Yolo predictor predicts single or multiple images with [darknet yolo](https://github.com/AlexeyAB/darknet) model
- Model has to be converted to tensorflow, you can use this repo for converting the model [github.com/hunglc007/tensorflow-yolov4-tflite](https://github.com/hunglc007/tensorflow-yolov4-tflite)
- Tested with tensorflow 2.4.0


## Usage
### Predict a single image
```shell
python predict.py -m model_weights_folder_path -n names_file_path -i image.jpg
```

### Predict a directory of images
```shell
python predict.py -m model_weights_folder_path -n names_file_path -d images_dir
```

### Predict with using yolo_predictor class
```python
from yolo_predictor import yolo_predictor

model_path = "model_files/model"
names_path = "model_files/model.names"
image_path = "test_images/image.jpg"

yp = yolo_predictor(model_path, names_path)

detection = yp.detect(image_path)
print(detection)
```

### Output
```shell
test_images/dog.jpg -> [('bicycle', 0.9867098, (0.4529685378074646, 0.48244842886924744, 0.573084, 0.5117002)), ('dog', 0.98514426, (0.28938552737236023, 0.6685629785060883, 0.23469335, 0.5305287)), ('truck', 0.92009175, (0.7547647655010223, 0.2147115096449852, 0.296138, 0.16953142))]
```