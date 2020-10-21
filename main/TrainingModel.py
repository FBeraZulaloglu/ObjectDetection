# python TrainingModel.py --dataset source\Objects --model source\OutputModels\objects.model --label-bin source\OutputLabels\objects_lb.pickle --plot source\OutputPlots\graphs.png

# import the necessary packages

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")# it enables us to save the plots to the disk
from source.SetModel.ClassModel import setModel
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset of images")
ap.add_argument("-m", "--model", required=True,
	help="path to output trained model")
ap.add_argument("-l", "--label-bin", required=True,
	help="path to output label binarizer")
ap.add_argument("-p", "--plot", required=True,
	help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

# initialize the data and labels
print("[INFO] loading images...")
data = [] #objects
labels = [] #object's names

if not os.path.exists(args["dataset"]):#["dataset"] is where all my objects images are
	print("The Training Data Set Is Not Available ! So I am not running")
	exit()

# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images(args["dataset"])))#imutils
random.seed(42)
random.shuffle(imagePaths)

# loop over the input images
for imagePath in imagePaths:
	# load the image, resize it to 64x64 pixels
	# then add the data list
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (64, 64))
	data.append(image)

	# extract the class label from the image path and update the
	# labels list
	label = imagePath.split(os.path.sep)[-2] # seperate to get the directory
	labels.append(label)

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)


(trainX, testX, trainY, testY) = train_test_split(data,labels, test_size=0.33,train_size=0.66, random_state=42)


lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,height_shift_range=0.1, shear_range=0.2,
						 zoom_range=0.2,horizontal_flip=True, fill_mode="nearest")

# initialize our VGG-like Convolutional Neural Network
# set model adds the properties to the model to take benefical resutls from it
model = setModel.buildModel(width=64, height=64, depth=3,classes=len(lb.classes_))

# initialize our initial learning rate, # of epochs to train for,
# and batch size
INIT_LR = 0.01
EPOCHS = 1000
BS = 32

# initialize the model and optimizer (you'll want to use
# binary_crossentropy for 2-class classification)
print("[INFO] training network...")
opt = SGD(lr=INIT_LR, decay=INIT_LR / EPOCHS)
#loss means error and when u try better your model you try to reduce loss
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the network
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY), steps_per_epoch=20, #len(trainX) // BS,
	epochs=EPOCHS)

# evaluate the network
print("[INFO] evaluating network...")
# predicts always take a list
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1), target_names=lb.classes_))
print("TEST X AND TEST Y :")
print(model.evaluate(testX,testY))

# plot the training loss and accuracy

# We are going to look here later
print(H.history.keys())
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.plot(N, H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["plot"])


print("[INFO] serializing network and label binarizer...")
# save the model and label binarizer to disk

model.save(args["model"])
print("The model has saved to the disk. WELL DONE YOU HAVE A BRAND-NEW MODEL")
f = open(args["label_bin"], "wb")
f.write(pickle.dumps(lb))

f.close()