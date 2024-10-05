import os
import sys
import time
import torch
import torch.nn

from utils import evaluate, get_dataset, FFDataset, setup_logger
from splits import split_data
from trainer import Trainer
import numpy as np
import random

#ignoring a futurewarning about torch.load having RCE(will be fixed by pytorch in future):
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

split_data()


# config
dataset_path = './../ff++'
pretrained_path = './../xception.pth'
batch_size = 16
gpu_ids = [0] #hard-coded to work only on the first available CUDA GPU, change to [] if no CUDA GPUs or update to more if needed
max_epoch = 5
loss_freq = 10
mode = 'Mix' # ['Original', 'FAD', 'LFS', 'Both', 'Mix']
ckpt_dir = './ckpts'
ckpt_name = 'cexxmix'
'''
NOTE: Modes:
         'Original': Uses only Xception
         'FAD': Only Frequency Aware Image Detection
         'LFS': Only uses Local Frequency Statistics
         'Both':  Uses both FAD and LFS and concatenates the results
         'Mix': Uses a cross attention model to combine the results of FAD and LFS
     Mix should give the best performance as per the paper
'''


if __name__ == '__main__':
    dataset = FFDataset(dataset_root=os.path.join(dataset_path, 'train', 'real'), size=299, frame_num=300, augment=True)
    dataloader_real = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size // 2,
        shuffle=True,
        num_workers=8
        )
    
    len_dataloader = dataloader_real.__len__()

    dataset_img, total_len =  get_dataset(name='train', size=299, root=dataset_path, frame_num=300, augment=True)
    dataloader_fake = torch.utils.data.DataLoader(
        dataset=dataset_img,
        batch_size=batch_size // 2,
        shuffle=True,
        num_workers=8
    )

    # init checkpoint and logger
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    logger = setup_logger(ckpt_path, 'context.log', 'logger')
    best_val = 0.
    ckpt_model_name = 'best.pkl'
    
    # train
    model = Trainer(gpu_ids, mode, pretrained_path)
    model.total_steps = 0
    epoch = 0
    
    while epoch < max_epoch:

        fake_iter = iter(dataloader_fake)
        real_iter = iter(dataloader_real)
        
        logger.debug(f'Epoch No. {epoch}')
        i = 0

        while i < len_dataloader:
            
            i += 1
            model.total_steps += 1

            try:
                data_real = next(real_iter)
                data_fake = next(fake_iter)
            except StopIteration:
                break
            # -------------------------------------------------
            
            if data_real.shape[0] != data_fake.shape[0]:
                continue

            bz = data_real.shape[0]
            
            data = torch.cat([data_real,data_fake],dim=0)
            label = torch.cat([torch.zeros(bz).unsqueeze(dim=0),torch.ones(bz).unsqueeze(dim=0)],dim=1).squeeze(dim=0)

            # manually shuffle
            idx = list(range(data.shape[0]))
            random.shuffle(idx)
            data = data[idx]
            label = label[idx]

            data = data.detach()
            label = label.detach()

            model.set_input(data,label)
            loss = model.optimize_weight()

            if model.total_steps % loss_freq == 0:
                logger.debug(f'loss: {loss} at step: {model.total_steps}')

            if i % int(len_dataloader / 10) == 0:
                model.model.eval()
                auc, r_acc, f_acc = evaluate(model, dataset_path, mode='valid')
                gold=0.4*auc+0.4*r_acc+0.2*f_acc #metric to measure the best model
                logger.debug(f'(Val @ epoch {epoch}) AUC: {auc}, r_acc: {r_acc}, f_acc:{f_acc}, score:{gold}')
                if gold > best_val:
                    best_val = gold
                    torch.save(model.model, os.path.join(ckpt_dir, f'best_{ckpt_name}.pkl'))
                    logger.info(f'Model saved at step: {model.total_steps} with score: {best_val}')
                auc, r_acc, f_acc = evaluate(model, dataset_path, mode='test')
                logger.debug(f'(Test @ epoch {epoch}) auc: {auc}, r_acc: {r_acc}, f_acc:{f_acc}')
                model.model.train()
        epoch = epoch + 1

    model.model.eval()
    auc, r_acc, f_acc = evaluate(model, dataset_path, mode='test')
    gold=0.4*auc+0.4*r_acc+0.2*f_acc
    logger.debug(f'(Test @ epoch {epoch}) AUC: {auc}, r_acc: {r_acc}, f_acc:{f_acc}, score:{gold}')
    torch.save(model.model, os.path.join(ckpt_dir, 'end.pkl'))
    logger.info('Final Model saved as end.pkl')
