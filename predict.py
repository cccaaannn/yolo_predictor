import os
import time
import argparse


# argparse type functions
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

def dim(d):
    try:
        d = int(d)
        if(d > 0):
            return d
        else:
            raise argparse.ArgumentTypeError()
    except:
        raise argparse.ArgumentTypeError("invalid image dim, Ex: 1280 720")


parser = argparse.ArgumentParser()

# model arguments
model_arguments_group = parser.add_argument_group("Model arguments")
model_arguments_group.add_argument("-m", '--model_path', dest="model_path", metavar="<path>", type=directory, required=True, help='Tensorflow converted saved model FOLDER path')
model_arguments_group.add_argument("-n", '--names_path', dest="names_path", metavar="<path>", type=path, required=True, help='Class (names) file path')

# image arguments
image_arguments_group = parser.add_argument_group("Image arguments")
mutually_exclusive_group = image_arguments_group.add_mutually_exclusive_group(required=True)
mutually_exclusive_group.add_argument("-d", '--dir_path', dest="dir_path", metavar="<path>", type=directory, help='Directory of the images')
mutually_exclusive_group.add_argument("-i", '--image_path', dest="image_path", metavar="<path>", type=path, help='Path of the image')

# draw arguments
draw_arguments_group = parser.add_argument_group("Draw arguments")
draw_arguments_group.add_argument("-s", '--show', dest="show", action="store_true", required=False, help='Draw and show images with bounding boxes')
draw_arguments_group.add_argument('--save_folder', dest="save_folder", metavar="<path>", type=directory, default=False, required=False, help='Save folder for drawn images with bounding boxes')
draw_arguments_group.add_argument('--resize', dest="resize", metavar="<dim>", type=dim, default=False, required=False, nargs=2, help='Resize images with given dim, Ex: 1280 720')
draw_arguments_group.add_argument('--suffix', dest="suffix", metavar="<string>", type=str, default="_predicted", required=False, help="Saved file suffix. Ex: '_predicted' dog.jpg -> dog_predicted.jpg")

args = parser.parse_args()



print("\n[Starting tensorflow]")
from yolo_predictor import yolo_predictor
from yolo_drawer import yolo_drawer


print("[Loading model]")
predictor = yolo_predictor(args.model_path, args.names_path)
drawer = yolo_drawer()


print("[Prediction started]\n")
# simgel image
if(args.image_path):
    # predict
    start = time.time()
    predictions = predictor.predict(args.image_path)
    end = time.time()

    current_prediction_time = end - start

    # print
    print("Image: {0}".format(args.image_path))
    print("Prediction: {0}".format(predictions))
    print("Time: {0:.2f}s\n".format(float(current_prediction_time)))

    # draw
    if(args.show or args.save_folder):
        drawer.draw(predictions, args.image_path, show = args.show, resize = args.resize, save_folder_path = args.save_folder, saved_file_suffix = args.suffix)

# multiple images
if(args.dir_path):
    total_time = 0
    for image_path in sorted(os.listdir(args.dir_path)):
        image_full_path = os.path.join(args.dir_path, image_path)

        try:
            # predict
            start = time.time()
            predictions = predictor.predict(image_full_path)
            end = time.time()

            current_prediction_time = end - start
            total_time += current_prediction_time

            # print
            print("Image: {0}".format(image_full_path))
            print("Prediction: {0}".format(predictions))
            print("Time: {0:.2f}s\n".format(float(current_prediction_time)))

            # draw
            if(args.show or args.save_folder):
                drawer.draw(predictions, image_full_path, show = args.show, resize = args.resize, save_folder_path = args.save_folder, saved_file_suffix = args.suffix)
        except:
            print("File '{0}' caused error skipping.\n".format(image_full_path))

    print("Total time: {0:.2f}s\n".format(float(total_time)))
