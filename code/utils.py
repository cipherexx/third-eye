import torch
import os
import numpy as np
import random
from torch.utils import data
from torchvision import transforms as trans
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc as cal_auc
from PIL import Image
import sys
import logging


class FFDataset(data.Dataset):

    def __init__(self, dataset_root, frame_num=300, size=299, augment=True):
        self.data_root = dataset_root
        self.frame_num = frame_num
        self.train_list = self.collect_image(self.data_root)
        if augment:
            self.transform = trans.Compose([trans.RandomHorizontalFlip(p=0.5), trans.ToTensor()])
            print("Augment True!")
        else:
            self.transform = trans.ToTensor()
        self.max_val = 1.
        self.min_val = -1.
        self.size = size

    def collect_image(self, root):
        image_path_list = []
        for split in os.listdir(root):
            split_root = os.path.join(root, split)
            img_list = os.listdir(split_root)
            random.shuffle(img_list)
            img_list = img_list if len(img_list) < self.frame_num else img_list[:self.frame_num]
            for img in img_list:
                img_path = os.path.join(split_root, img)
                image_path_list.append(img_path)
        return image_path_list

    def read_image(self, path):
        img = Image.open(path)
        return img

    def resize_image(self, image, size):
        img = image.resize((size, size))
        return img

    def __getitem__(self, index):
        image_path = self.train_list[index]
        img = self.read_image(image_path)
        img = self.resize_image(img,size=self.size)
        img = self.transform(img)
        img = img * (self.max_val - self.min_val) + self.min_val
        return img

    def __len__(self):
        return len(self.train_list)


def get_dataset(name = 'train', size=299, root='./ff++', frame_num=300, augment=True):
    root = os.path.join(root, name)
    fake_root = os.path.join(root,'fake')

    fake_list = ['Deepfakes', 'Face2Face']
    
    total_len = len(fake_list)
    dset_lst = []
    for i in range(total_len):
        fake = os.path.join(fake_root , fake_list[i])
        dset = FFDataset(fake, frame_num, size, augment)
        dset.size = size
        dset_lst.append(dset)
    return torch.utils.data.ConcatDataset(dset_lst), total_len

def evaluate(model, data_path, mode='valid'):
    root= data_path
    origin_root = root
    root = os.path.join(data_path, mode)
    real_root = os.path.join(root,'real')
    dataset_real = FFDataset(dataset_root=real_root, size=299, frame_num=50, augment=False)
    dataset_fake, _ = get_dataset(name=mode, root=origin_root, size=299, frame_num=50, augment=False)
    dataset_img = torch.utils.data.ConcatDataset([dataset_real, dataset_fake])

    bz = 64
    with torch.no_grad():
        y_true, y_pred = [], []

        for i, d in enumerate(dataset_img.datasets):
            dataloader = torch.utils.data.DataLoader(
                dataset = d,
                batch_size = bz,
                shuffle = True,
                num_workers = 8
            )
            for img in dataloader:
                if i == 0:
                    label = torch.zeros(img.size(0))
                else:
                    label = torch.ones(img.size(0))
                #img=img.detach().cuda()
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                img = img.detach().to(device)

                output = model.forward(img)
                y_pred.extend(output.sigmoid().flatten().tolist())
                y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    fpr, tpr, thresholds = roc_curve(y_true,y_pred,pos_label=1)
    AUC = cal_auc(fpr, tpr)

    idx_real = np.where(y_true==0)[0]
    idx_fake = np.where(y_true==1)[0]

    r_acc = accuracy_score(y_true[idx_real], y_pred[idx_real] > 0.5)
    f_acc = accuracy_score(y_true[idx_fake], y_pred[idx_fake] > 0.5)

    return AUC, r_acc, f_acc


__all__ = ['setup_logger']

DEFAULT_WORK_DIR = 'output' #all files created during the working of the program are saved here

def setup_logger(work_dir=None, logfile_name='log.txt', logger_name='logger'):
    """
    This function sets up a logger to be used throughout. All messages are prefixed by current time and level.
    All messages are outputted to the given logfile, and messages with INFO or above level are
    also outputted to the console.
    NOTE: If logfile_name is empty, logging to file is skipped
    """

    logger = logging.getLogger(logger_name)
    if logger.hasHandlers():  # Logger with this name has already existed
        raise SystemExit(f'A logger with the name`{logger_name}` has already been set up\n'
                            f'Please use another name, otherwise the messages '
                            f'may be mixed between these two loggers.')

    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")

    # Print log message with INFO level or above onto the screen:
    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    if not logfile_name:
        return logger

    work_dir = work_dir or DEFAULT_WORK_DIR
    logfile_name = os.path.join(work_dir, logfile_name)

    os.makedirs(work_dir, exist_ok=True)

    # Save log messages to file:
    fh = logging.FileHandler(logfile_name, mode="a")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
