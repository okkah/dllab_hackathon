import numpy as np
import os
from PIL import Image
import pickle

train_samples = 100
img_shape=(224,224)

imgs = []

with open("../testdata.pickle", "rb") as f:
    test = pickle.load(f)

with open("../traindata.pickle", "rb") as f:
    train = pickle.load(f)

"""
for filename in os.listdir("../data/train/images")[:train_samples]:
img = Image.open("../data/train/images/" + filename)
img = img.resize(img_shape)
img = np.asarray(img, dtype='f')
img /= 255
img = np.transpose(img, (2, 0, 1))
imgs.append((filename, img))
"""
    
labels = {}
with open("../data/train/train_labels.txt", "r") as f:
    for line in f:
        labels[line.split()[0]] = line.split()[1]    

"""
train = []
for fn, img in imgs:
 train.append((img, np.int8(labels[fn])))
"""

train_samples = 50
img_shape=(224,224)

imgs = []
for filename in os.listdir("../data/valid/images")[:train_samples]:
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


import chainer
import chainer.links as L
import chainer.functions as F

class BatchConv2D(chainer.Chain):
    def __init__(self, ch_in, ch_out, ksize, stride=1, pad=0, activation=F.relu):
        super(BatchConv2D, self).__init__(
            conv=L.Convolution2D(ch_in, ch_out, ksize, stride, pad),
            bn=L.BatchNormalization(ch_out),
        )
        self.activation=activation

    def __call__(self, x):
        h = self.bn(self.conv(x))
        if self.activation is None:
            return h
        return F.relu(h)

class VGG(chainer.Chain):

    def __init__(self, n_mid=100, n_out=9):
       super(VGG, self).__init__(
            bconv1_1=BatchConv2D(None, 64, ksize=3, stride=1, pad=1),
            bconv1_2=BatchConv2D(None, 64, ksize=3, stride=1, pad=1),
            bconv2_1=BatchConv2D(None, 128, ksize=3, stride=1, pad=1),
            bconv2_2=BatchConv2D(None, 128, ksize=3, stride=1, pad=1),
            bconv3_1=BatchConv2D(None, 256, ksize=3, stride=1, pad=1),
            bconv3_2=BatchConv2D(None, 256, ksize=3, stride=1, pad=1),
            bconv3_3=BatchConv2D(None, 256, ksize=3, stride=1, pad=1),
            bconv3_4=BatchConv2D(None, 256, ksize=3, stride=1, pad=1),
            fc4=L.Linear(None, 1024),
            fc5=L.Linear(None, 1024),
            fc6=L.Linear(None, n_out),
      )

    def __call__(self, x):
        h = self.bconv1_1(x)
        h = self.bconv1_2(h)
        h = F.dropout(F.max_pooling_2d(h, 2), 0.25)
        h = self.bconv2_1(h)
        h = self.bconv2_2(h)
        h = F.dropout(F.max_pooling_2d(h, 2), 0.25)
        h = self.bconv3_1(h)
        h = self.bconv3_2(h)
        h = self.bconv3_3(h)
        h = self.bconv3_4(h)
        h = F.dropout(F.max_pooling_2d(h, 2), 0.25)
        h = F.relu(self.fc4(F.dropout(h)))
        h = F.relu(self.fc5(F.dropout(h)))
        return self.fc6(h)
        
import random

def reset_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    if chainer.cuda.available:
        chainer.cuda.cupy.random.seed(seed)


reset_seed(0)

model = L.Classifier(VGG())

gpu_id = 0  # 使用したGPUに割り振られているID
model.to_gpu(gpu_id)


optimizer = chainer.optimizers.AdaGrad()
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.WeightDecay(0.0005))

batchsize = 30
train_iter = chainer.iterators.SerialIterator(train, batchsize)
test_iter = chainer.iterators.SerialIterator(test, batchsize, repeat=False, shuffle=False)


from chainer import training
from chainer.training import extensions

epoch = 60

updater = training.StandardUpdater(train_iter, optimizer, device=gpu_id)

trainer = training.Trainer(updater, (epoch, 'epoch'), out='mnist')

# バリデーション用のデータで評価
trainer.extend(extensions.Evaluator(test_iter, model, device=gpu_id))

# 学習結果の途中を表示する
trainer.extend(extensions.LogReport(trigger=(1, 'epoch')))

# １エポックごとに結果をlogファイルに出力させる
trainer.extend(extensions.PrintReport(['epoch', 'main/accuracy', 'validation/main/accuracy', 'main/loss', 'validation/main/loss', 'elapsed_time']), trigger=(1, 'epoch'))

trainer.run()
