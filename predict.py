import os
import argparse

from yolo_predictor import yolo_predictor


def directory(p):
    if(os.path.isdir(p)):
        return p
    else:
        raise argparse.ArgumentTypeError("directory does not exists")

def path(p):
    if(os.path.isfile(p)):
        return p
    else:
        raise argparse.ArgumentTypeError("file does not exists")

parser = argparse.ArgumentParser()

# Add the arguments
parser.add_argument("-m", '--model_path', dest="model_path", type=directory, help='tensorflow converted saved model FOLDER path', required=True)
parser.add_argument("-n", '--names_path', dest="names_path", type=path, help='class file path', required=True)

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("-d", '--dir_path', dest="dir_path", type=directory, help='directory of the images')
group.add_argument("-i", '--image_path', dest="image_path", type=path, help='path of the image')

args = parser.parse_args()

model_path = args.model_path
names_path = args.names_path
image_path = args.image_path
dir_path = args.dir_path



print("Loading model...")
yp = yolo_predictor(model_path, names_path)

print("Prediction started\n")
if(image_path):
    detection = yp.detect(image_path)
    print("{0} -> {1}".format(image_path, detection))
if(dir_path):
    for image_path in sorted(os.listdir(dir_path)):
        detection = yp.detect(os.path.join(dir_path, image_path))
        print("{0} -> {1}".format(image_path, detection))

