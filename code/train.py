# -*- coding: utf-8 -*-
import os
import json
import random

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data

from config import opt
from utils import non_model, in_model
from make_dataset import train_Dataset, val_Dataset
from net import model_TransSR

import numpy as np

from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2000, rlimit[1]))

def train(**kwargs):
    # region
    GLOBAL_SEED = 0
    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)
    torch.cuda.manual_seed(GLOBAL_SEED)
    torch.cuda.manual_seed_all(GLOBAL_SEED)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    GLOBAL_WORKER_ID = None

    def worker_init_fn(worker_id):
        global GLOBAL_WORKER_ID
        GLOBAL_WORKER_ID = worker_id
        set_seed(GLOBAL_SEED + worker_id)
    # endregion

    kwargs, data_info_dict = non_model.read_kwargs(kwargs)
    opt.load_config('../config/all.txt')
    config_dict = opt._spec(kwargs)

    # Path
    save_model_folder = '../model/%s/' % (str(opt.net_idx))
    save_info_folder = '../info/%s/' % (str(opt.net_idx))
    os.makedirs(save_model_folder, exist_ok=True)
    os.makedirs(save_info_folder, exist_ok=True)
    with open(save_info_folder + 'config.json', 'w', encoding='utf-8') as json_file:
        json.dump(config_dict, json_file, ensure_ascii=False, indent=4)

    # Network
    data_gpu = opt.gpu_idx
    torch.cuda.set_device(data_gpu)

    net = model_TransSR.TVSRN3d_v1()
    net = net.cuda()

    # Optim
    print('================== AdamW lr = %.6f ==================' % opt.lr)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()),
                                    lr=opt.lr, weight_decay=opt.wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=opt.patience, threshold=0.000001)

    # Loss
    train_criterion = nn.L1Loss()
    val_criterion = nn.L1Loss()

    # Dataloader
    train_list = data_info_dict['train']
    val_list = data_info_dict['val']

    train_set = train_Dataset(train_list)
    train_data_num = len(train_set.img_list)
    train_batch = Data.DataLoader(dataset=train_set, batch_size=opt.train_bs, shuffle=True, \
                                  num_workers=opt.num_workers, worker_init_fn=worker_init_fn, \
                                  drop_last=True)
    print('load train data done, num =', train_data_num)

    val_set = val_Dataset(val_list)
    val_data_num = len(val_set.img_list)
    val_batch = Data.DataLoader(dataset=val_set, batch_size=opt.val_bs, shuffle=False,
                                num_workers=opt.test_num_workers, worker_init_fn=worker_init_fn)
    print('load val data done, num =', val_data_num)

    ###### Start Training ######
    epoch_save = 0
    lr_change = 0
    best_metric = 0

    for e in range(opt.epoch):
        tmp_epoch = e+opt.start_epoch
        tmp_lr = optimizer.__getstate__()['param_groups'][0]['lr']
        print('================= Epoch %s lr=%.6f =================' % (tmp_epoch, tmp_lr))

        if lr_change == 4:
            break

        # Train
        train_loss = 0
        net = net.train()
        skip_val = False

        for i, return_list in tqdm(enumerate(train_batch)):
            x, y = return_list
            x = Variable(x.type(torch.FloatTensor).cuda())
            label = Variable(y.type(torch.FloatTensor).cuda())

            y_pre = net(x)
            loss = train_criterion(y_pre, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            del y_pre, label, x

        torch.cuda.empty_cache()
        train_loss = train_loss / len(train_batch)

        # gap val
        if train_loss < 0.01:
            skip_val = True
        
        if opt.gap_val != 0 and e % opt.gap_val != 0:
            print('epoch %s, train_loss: %.4f' % (tmp_epoch, train_loss))
            continue
        elif skip_val == False:
            print('epoch %s, train_loss: %.4f' % (tmp_epoch, train_loss))

            scheduler.step(train_loss)
            before_lr = optimizer.__getstate__()['param_groups'][0]['lr']
            if before_lr != tmp_lr:
                lr_change += 1
            continue

        net = net.eval()
        with torch.no_grad():
            psnr_list = []

            for i, return_list in tqdm(enumerate(val_batch)):
                case_name, x, y, pos_list = return_list
                case_name = case_name[0]
                x = x.squeeze().data.numpy()
                y = y.squeeze().data.numpy()

                if e == 0 and i == 0:
                    print('thin size:', y.shape)

                y_pre = np.zeros_like(y)
                pos_list = pos_list.data.numpy()[0]

                for pos_idx, pos in enumerate(pos_list):
                    tmp_x = x[pos_idx]
                    tmp_pos_z, tmp_pos_y, tmp_pos_x = pos

                    tmp_x = torch.from_numpy(tmp_x)
                    tmp_x = tmp_x.unsqueeze(0).unsqueeze(0)
                    im = Variable(tmp_x.type(torch.FloatTensor).cuda())
                    tmp_y_pre = net(im)
                    tmp_y_pre = torch.clamp(tmp_y_pre, 0, 1)
                    y_for_psnr = tmp_y_pre.data.squeeze().cpu().numpy()

                    D = y_for_psnr.shape[0]
                    pos_z_s = 5 * tmp_pos_z + 3
                    pos_y_s = tmp_pos_y
                    pos_x_s = tmp_pos_x

                    y_pre[pos_z_s: pos_z_s+D, pos_y_s:pos_y_s+opt.vc_y, pos_x_s:pos_x_s+opt.vc_x] = y_for_psnr

                del tmp_y_pre, im

                y_pre = y_pre[5:-5]
                y = y[5:-5]

                psnr = non_model.cal_psnr(y_pre, y)
                psnr_list.append(psnr)

        torch.cuda.empty_cache()

        psnr_val = np.array(psnr_list).mean()
        print('epoch %s, train_loss: %.4f, psnr_val:, %.4f'%(tmp_epoch, train_loss, psnr_val))

        if psnr_val > best_metric:
            best_metric = psnr_val
            epoch_save = tmp_epoch
            save_dict = {}
            save_dict['net'] = net
            save_dict['config_dict'] = config_dict
            torch.save(save_dict, save_model_folder + '%s_Loss_%.4f_PSNR_%.4f.pkl' %
                           (str(tmp_epoch).rjust(4,'0'), train_loss, psnr_val))

            del save_dict
            print('====================== model save ========================')

        scheduler.step(train_loss)

        before_lr = optimizer.__getstate__()['param_groups'][0]['lr']
        if before_lr != tmp_lr:
            lr_change += 1

        torch.cuda.empty_cache()

if __name__ == '__main__':
    import fire

    fire.Fire()
    

