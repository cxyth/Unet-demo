#coding=utf-8
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.utils import to_categorical  
from tensorflow.keras.preprocessing.image import img_to_array  
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import multi_gpu_model, plot_model
from sklearn.preprocessing import LabelEncoder  
from tensorflow.keras.models import Model
#from PIL import Image  
import matplotlib.pyplot as plt  
import cv2
import random
import os
from tqdm import tqdm  
#os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2"
seed = 7  
np.random.seed(seed)  
  
#data_shape = 360*480  
img_w = 512
img_h = 512
GRAY = True
n_label = 1+2
  
classes = [0., 1., 2.]  
  
labelencoder = LabelEncoder()  
labelencoder.fit(classes)  
 

def get_train_val(train_dir, valid_dir):   
    train_set = []
    valid_set  = []
    for pic in os.listdir(train_dir + 'image/'):
        data = ((train_dir+'image/'+pic), (train_dir+'label/'+pic))
        train_set.append(data)
    for pic in os.listdir(valid_dir + 'image/'):
        data = ((valid_dir+'image/'+pic), (valid_dir+'label/'+pic))
        valid_set.append(data)
    return train_set,valid_set

def get_random_data(image, mask):
    '''random preprocessing for real-time data augmentation'''
    
    def rotate(xb, yb, angle):
        M_rotate = cv2.getRotationMatrix2D((img_w / 2, img_h / 2), angle, 1)
        xb = cv2.warpAffine(xb, M_rotate, (img_w, img_h))
        yb = cv2.warpAffine(yb, M_rotate, (img_w, img_h))
        return xb, yb

    def random_augment(xb, yb):
        xb = xb.astype(np.uint8)
        r = np.random.random()
        if r < 0.25:
            xb, yb = rotate(xb, yb, 90)
        elif (r >= 0.25) and (r < 0.5):
            xb, yb = rotate(xb, yb, 180)
        elif (r >= 0.5) and (r < 0.75):
            xb, yb = rotate(xb, yb, 270)
        # else: do nothing
        r = np.random.random()
        if r < 0.25:
            # Flipped Horizontally 水平翻转
            xb = cv2.flip(xb, 1)
            yb = cv2.flip(yb, 1)
        elif (r >= 0.25) and (r < 0.5):
            # Flipped Vertically 垂直翻转
            xb = cv2.flip(xb, 0)
            yb = cv2.flip(yb, 0)
        elif (r >= 0.5) and (r < 0.75):
            # Flipped Horizontally & Vertically 水平垂直翻转
            xb = cv2.flip(xb, -1)
            yb = cv2.flip(yb, -1)
        return xb, yb

    image, mask = random_augment(image, mask)
    return image, mask
              
                
def generateData(batch_size, data, random_aug=False):  
    #print 'generateValidData...'
    while True:  
        img_data = []  
        label_data = []  
        batch = 0  
        for i in (range(len(data))):  
            url = data[i]
            batch += 1  
            img = cv2.imread(url[0],cv2.IMREAD_GRAYSCALE)
            label = cv2.imread(url[1],cv2.IMREAD_GRAYSCALE)
            if random_aug:
                img, label = get_random_data(img, label)
            img = img_to_array(img) / 255.0
            img_data.append(img)  
            label = img_to_array(label).reshape((img_w * img_h,))  
            label_data.append(label)  
            if batch % batch_size==0:  
                img_data = np.array(img_data)  
                label_data = np.array(label_data).flatten()  
                label_data = labelencoder.transform(label_data)  
                label_data = to_categorical(label_data, num_classes=n_label)  
                label_data = label_data.reshape((batch_size,img_h*img_w, n_label))
                yield (img_data,label_data)  
                img_data = []  
                label_data = []  
                batch = 0 


#自定义评价指标
def iou(y_true, y_pred, label: int):
    """
    Return the Intersection over Union (IoU) for a given label.
    Args:
        y_true: the expected y values as a one-hot
        y_pred: the predicted y values as a one-hot or softmax output
        label: the label to return the IoU for
    Returns:
        the IoU for the given label
    """
    # extract the label values using the argmax operator then
    # calculate equality of the predictions and truths to the label
    y_true = K.cast(K.equal(K.argmax(y_true), label), K.floatx())
    y_pred = K.cast(K.equal(K.argmax(y_pred), label), K.floatx())
    # calculate the |intersection| (AND) of the labels
    intersection = K.sum(y_true * y_pred)
    # calculate the |union| (OR) of the labels
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    # avoid divide by zero - if the union is zero, return 1
    # otherwise, return the intersection over union
    return K.switch(K.equal(union, 0), 1.0, intersection / union)


def miou(y_true, y_pred):
    """
    Return the Intersection over Union (IoU) score.
    Args:
        y_true: the expected y values as a one-hot
        y_pred: the predicted y values as a one-hot or softmax output
    Returns:
        the scalar IoU value (mean over all labels)
    """
    # get number of labels to calculate IoU for
    num_labels = K.int_shape(y_pred)[-1] - 1
    # initialize a variable to store total IoU in
    mean_iou = K.variable(0)

    # iterate over labels to calculate IoU for
    for label in range(num_labels+1):
        mean_iou = mean_iou + iou(y_true, y_pred, label)

    # divide total IoU by number of labels to get mean IoU
    return mean_iou / num_labels


#自定义回调函数，保存模型
class ParallelModelCheckpoint(ModelCheckpoint):
    def __init__(self,model,filepath,monitor='val_acc',
                 save_best_only=True,mode='max'):
        self.single_model = model
        super(ParallelModelCheckpoint,self).__init__(filepath,monitor,save_best_only,mode)

    def set_model(self, model):
        super(ParallelModelCheckpoint,self).set_model(self.single_model)

class BackupModelCheckpoint(ModelCheckpoint):
    def __init__(self,model,filepath):
        self.single_model = model
        super(BackupModelCheckpoint,self).__init__(filepath)

    def set_model(self, model):
        super(BackupModelCheckpoint,self).set_model(self.single_model)

  
def train(args): 
    EPOCHS = args['epochs']
    BS = args['batch_size']
    original_model = args['model']
    gpus = args['gpus']
    if gpus > 1:
        train_model = multi_gpu_model(original_model, gpus = gpus)
    else:
        train_model = original_model
    train_model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['acc',miou])
    modelcheck = ParallelModelCheckpoint(original_model, args['save_path'],
			monitor='val_miou',save_best_only=True,mode='max')  
    #backup_path = 'backup/model-{epoch:02d}-{val_loss:.2f}.h5'
    backup_path = 'backup/model-{epoch:02d}.h5'
    modelbackup = BackupModelCheckpoint(original_model,backup_path)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1, min_lr=0.00001)#factor:reduce rate
    #callable = [modelcheck,modelbackup,reduce_lr]
    callable = [modelcheck, reduce_lr]
    train_set,val_set = get_train_val(train_dir=args['train_dir'], valid_dir=args['valid_dir'])
    #print(train_set[:5])
    #print(val_set[:5])
    train_numb = len(train_set)  
    valid_numb = len(val_set)  
    print ("the number of train data is",train_numb)  
    print ("the number of val data is",valid_numb)
    H = train_model.fit_generator(generator=generateData(BS,train_set,random_aug=True),steps_per_epoch=train_numb//BS,epochs=EPOCHS,verbose=1,
                    validation_data=generateData(BS,val_set,random_aug=False),validation_steps=valid_numb//BS,callbacks=callable)  

    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    N = EPOCHS
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on U-Net Seg")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig("plot.png")

  


if __name__=='__main__':  
    # test code

    '''
    from backup.model import unet
    
    unet_model = unet(input_shape=(512, 512, 1), n_label=3)
    unet_model.summary()
    plot_model(unet_model, to_file='unet_model.jpg')
    
    args = dict(
        model=unet_model,
        epochs=3,
        batch_size=1, #16
        gpus=1, 
        train_dir='./dataset/split_512/apport/train/',
        valid_dir='./dataset/split_512/apport/valid/',
        save_path='./Unet.h5'
    )
    train(args)    
    '''

