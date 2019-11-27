import cv2
import os
import numpy as np
import random
import shutil
from tqdm import tqdm


def get_filelist(path, ext=[]):
    file_list = []
    files = os.listdir(path)
    for f in files:
        if f.split('.')[-1] in ext:
            file_list.append(f)
    return file_list

def load_img(path, grayscale=False):
    if grayscale:
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path)
    img = np.array(img,dtype="float") / 255.0
    return img

def load_label(path):
    return cv2.imread(path,cv2.IMREAD_GRAYSCALE)

def get_random_data(annotation, random=True, grayscale = False):
    '''random preprocessing for real-time data augmentation'''
    images = load_img(train_path + 'src/' + annotation, grayscale = grayscale)
    mask = load_label(train_path + 'label/' + annotation)
    if not random:
        return np.array(images), mask
    img_h, img_w, _ = images.shape
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
        # else: 啥也不干

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

    images, mask = random_augment(images, mask)
    return np.array(images), mask


def split_img_and_mask(image, label, split_size, split_dir):
    count = 0
    split_img_path = os.path.join(split_dir, 'image')
    if not os.path.exists(split_img_path):
        os.makedirs(split_img_path)
        print('makedir: ', split_img_path)
    split_label_path = os.path.join(split_dir, 'label')
    if not os.path.exists(split_label_path):
        os.makedirs(split_label_path)
        print('makedir: ', split_label_path)

    src_label = label
    src_img = image
    src_h,src_w,src_c = src_img.shape

    for i in range(int(src_w/split_size)):
        w = i*split_size
        for j in range(int(src_h/split_size)):
            h = j*split_size
            
            src_out = src_img[h:h+split_size,w:w+split_size]
            label_out = src_label[h:h+split_size,w:w+split_size]

            #检查子图的空白占比
            flag = (src_out == 0)
            flag = np.all(flag, axis=-1)
            flag = flag.astype(np.int)
            if np.sum(flag) < (split_size*split_size*0.4):
                cv2.imwrite(os.path.join(split_img_path, 'img_' + str(w) + '_' + str(h) + '.png'), src_out)
                cv2.imwrite(os.path.join(split_label_path, 'img_' + str(w) + '_' + str(h) + '.png'), label_out)
                print('{:4d} write: {}'.format(count+1, 'img_' + str(w) + '_' + str(h) + '.png'))
                count += 1
    print('done!')



def apportion(src_dir, apport_dir, apport_rate, seed):
    #建立目录
    train_src_path = os.path.join(apport_dir, 'train','image')
    train_label_path = os.path.join(apport_dir, 'train','label')
    valid_src_path = os.path.join(apport_dir, 'valid','image')
    valid_label_path = os.path.join(apport_dir, 'valid','label')
    if not os.path.exists(train_src_path):
        os.makedirs(train_src_path)
    if not os.path.exists(train_label_path):
        os.makedirs(train_label_path)      
    if not os.path.exists(valid_src_path):
        os.makedirs(valid_src_path)
    if not os.path.exists(valid_label_path):
        os.makedirs(valid_label_path)        
    
    src_path = os.path.join(src_dir, 'image')
    label_path = os.path.join(src_dir, 'label')
    all_imgs = get_filelist(src_path,['png'])

    train_corn = 0
    train_baccy = 0
    valid_corn = 0
    valid_baccy = 0
    train_list = []
    valid_list = []
    total_num = len(all_imgs)
    train_num = total_num*apport_rate['train']
    valid_num = total_num*apport_rate['valid']
    print('train:%d | valid:%d '%(train_num,valid_num))
    
    random.seed(seed)
    random.shuffle(all_imgs)
    for i in range(total_num):
        if i <train_num:
            train_list.append(all_imgs[i])
        elif (i >= train_num) and (i < total_num):
            valid_list.append(all_imgs[i])
    
    for f in train_list:
        mask = cv2.imread(label_path+'/'+f, -1)
        if np.any(mask==1):
            train_corn += 1
        if np.any(mask==2):
            train_baccy += 1
        shutil.copy(src_path+'/'+f, train_src_path)
        shutil.copy(label_path+'/'+f, train_label_path)
    for f in valid_list:
        mask = cv2.imread(label_path+'/'+f, -1)
        if np.any(mask==1):
            valid_corn += 1
        if np.any(mask==2):
            valid_baccy += 1
        shutil.copy(src_path+'/'+f, valid_src_path)
        shutil.copy(label_path+'/'+f, valid_label_path)

    if (train_num > 0):
        print('train_corn: {:.2f}%, train_baccy: {:.2f}%'.format(train_corn / train_num*100, train_baccy / train_num*100))
    if (valid_num > 0):
        print('valid_corn: {:.2f}%, valid_baccy: {:.2f}%'.format(valid_corn / valid_num*100, valid_baccy / valid_num*100))
    print('done!')

                   
if __name__ == '__main__':
    src_dir = './dataset/split_512/'
    apport_dir = './dataset/split_512/apport/'
    # train/valid 分配比例
    apport_rate = {'train':0.70,'valid':0.30}
    apportion(src_dir, apport_dir, apport_rate)