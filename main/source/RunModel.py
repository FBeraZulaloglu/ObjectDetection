# USAGE
# python predict.py --image images/dog.jpg --model output/simple_nn.model --label-bin output/simple_nn_lb.pickle --width 32 --height 32 --flatten 1
# python predict.py --image images/dog.jpg --model output/smallvggnet.model --label-bin output/smallvggnet_lb.pickle --width 64 --height 64

# import the necessary packages
from keras.models import load_model
import pickle
import cv2
import os
import numpy as np
import cvlib
from cvlib.object_detection import draw_bbox
import cv2


class Predict:
    def predictImage(self, imagePath):

        modelPath = r"C:\Users\faruk\PycharmProjects\ObjectDetectionProject\my_object.model"
        lbPath = r"C:\Users\faruk\PycharmProjects\ObjectDetectionProject\my_object_lb.pickle"

        if not os.path.exists(modelPath):
            print("Model does not exists")
            exit()
        if not os.path.exists(lbPath):
            print("Label Binarizer Does Not Exists")
            exit()
        if not os.path.exists(imagePath):
            print("The image does not exists")
            exit()
        # load the input image and resize it to the target spatial dimensions
        image = cv2.imread(imagePath)
        #output = image
        image = cv2.resize(image,(64,64))

        # scale the pixel values to [0, 1]
        image = image.astype("float") / 255.0

        # check to see if we should flatten the image and add a batch
        # dimension

        #image = image.flatten()
        #image = image.reshape((-1, image.shape[0]))

        # otherwise, we must be working with a CNN -- don't flatten the
        # image, simply add the batch dimension
        image = image.reshape((1, image.shape[0], image.shape[1],image.shape[2]))

        # load the model and label binarizer
        print("[INFO] loading network and label binarizer...")
        model = load_model(modelPath)
        lb = pickle.loads(open(lbPath, "rb").read())

        # make a prediction on the image
        preds = model.predict(image)
        

        # find the class label index with the largest corresponding
        # probability
        i = preds.argmax(axis=1)[0]
        label = lb.classes_



        # draw the class label + probability on the output image
        text = "{}: {:.2f}%".format(label[i], preds[0][i] * 100)
        print(text)
        # cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0, 0, 255), 2)
        print("PREDİCT HAS ACCOMPLISHED")
        return preds,label



p = Predict()
image_path = r"C:\Users\faruk\PycharmProjects\ObjectDetectionProject\source\Objects\apple\Apple-cube-1.jpg"
p.predictImage(image_path)