from PIL import Image
import os
import numpy as np
import pickle

# train
img_shape=(224,224)
imgs = []
for filename in os.listdir("../data/train/images"):
    img = Image.open("../data/train/images/" + filename)
    img = img.resize(img_shape)
    img = np.asarray(img, dtype='f')
    img /= 255
    img = np.transpose(img, (2, 0, 1))
    imgs.append((filename, img))
    
labels = {}
with open("../data/train/train_labels.txt", "r") as f:
    for line in f:
        labels[line.split()[0]] = line.split()[1]    

train = []
for fn, img in imgs:
    train.append((img, np.int8(labels[fn])))

with open("traindata.pickle", "wb") as f:
    pickle.dump(train, f)


# test
img_shape=(224, 224)
imgs = []
for filename in os.listdir("../data/valid/images"):
    img = Image.open("../data/valid/images/" + filename)
    img = img.resize(img_shape)
    img = np.asarray(img, dtype='f')
    img /= 255
    img = np.transpose(img, (2, 0, 1))
    imgs.append((filename, img))
    
labels = {}
with open("../data/valid/valid_labels.txt", "r") as f:
    for line in f:
        labels[line.split()[0]] = line.split()[1]    

test = []
for fn, img in imgs:
    test.append((img, np.int8(labels[fn])))

with open("../testdata.pickle", "wb") as f:
    pickle.dump(test, f)


