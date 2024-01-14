import warnings
from helper_tool import Configcarla as cfg
from Network import Network, compute_loss, compute_acc,compute_loss1, IoUCalculator
from carla_dataset import carla, carlaSampler
import numpy as np
import os, argparse
import random
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import time
from utils.optim import create_optimizer_v2
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from augnet import augnet

seed=3407
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed) # 为CPU设置随机种子
torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.	
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = False        # 禁止cudnn加速，不加上反向传播会报错(数据矩阵太大了，如果用一个GPU的话)

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')
parser.add_argument('--log_dir', default='train_output_carla', help='Dump dir to save model checkpoint [default: log]')
parser.add_argument('--max_epoch', type=int, default=15, help='Epoch to run [default: 100]')    # 50够了
parser.add_argument('--gpu', type=int, default=0, help='which gpu do you want to use [default: 2], -1 for cpu')
parser.add_argument('--test_area', type=int, default='5', help='Which area to use for test, option: 1-6 [default: 6]')
parser.add_argument('--real_data', type=str, default=True, help='Which area to use for real data train')
FLAGS = parser.parse_args()

#################################################   log   #################################################
LOG_DIR = FLAGS.log_dir
LOG_DIR = os.path.join(LOG_DIR, time.strftime('%Y-%m-%d_%H-%M-%S', time.gmtime()))      # 返回的是英国时间
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)                # 创建多级目录
log_file_name = f'log_train_carla_{FLAGS.test_area:d}.txt'
LOG_FOUT = open(os.path.join(LOG_DIR, log_file_name), 'a')      # 追加写入模式

def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)

#################################################   dataset   #################################################
dataset = carla()
training_dataset = carlaSampler(dataset, 'training')
validation_dataset = carlaSampler(dataset, 'validation')
training_dataloader= DataLoader(training_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=training_dataset.collate_fn,num_workers=2)
validation_dataloader = DataLoader(validation_dataset, batch_size=cfg.val_batch_size, shuffle=True, collate_fn=validation_dataset.collate_fn,num_workers=2)
if FLAGS.real_data:
    real_dataset = carlaSampler(dataset, 'real')
    real_dataloader= DataLoader(real_dataset, batch_size=1, shuffle=True, collate_fn=real_dataset.collate_fn,num_workers=2)
    print(len(training_dataloader), len(validation_dataloader),len(real_dataloader))
else:
    print(len(training_dataloader), len(validation_dataloader))
#################################################   network   #################################################

if FLAGS.gpu >= 0:
    if torch.cuda.is_available():
        FLAGS.gpu = torch.device(f'cuda:{FLAGS.gpu:d}')
    else:
        warnings.warn('CUDA is not available on your machine. Running the algorithm on CPU.')
        FLAGS.gpu = torch.device('cpu')
else:
    FLAGS.gpu = torch.device('cpu')

device = FLAGS.gpu
net = Network(cfg)
if FLAGS.real_data:
    aug = augnet()
    aug.to(device)
net.to(device)

# Load the Adam optimizer
optimizer = optim.Adam(net.parameters(), lr=cfg.learning_rate)
start_epoch = 0
CHECKPOINT_PATH = FLAGS.checkpoint_path
if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    log_string("-> loaded checkpoint %s (epoch: %d)"%(CHECKPOINT_PATH, start_epoch))

#################################################   training functions   ###########################################
def adjust_learning_rate(optimizer, epoch):
    lr = optimizer.param_groups[0]['lr']        # param_groups 是一个长度为1的列表（可能有时不为一），列表里面是字典，字典中有该优化器相关的参数
    lr = lr * cfg.lr_decays[epoch]              # cfg.lr_decays一个有500个键值对的字典，每个键对应的值都是0.95，也就是每个epoch学习率衰减0.95
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr                  # 赋值新的学习率

def train_one_epoch():
    stat_dict = {}  # collect statistics
    adjust_learning_rate(optimizer, EPOCH_CNT)
    net.train()  # set model to training mode
    iou_calc = IoUCalculator(cfg)               # 初始化IOU计算器
    train_loss = 0
    train_accuracy = 0
    R_list,scale_list,noise_list = [],[],[]
    if FLAGS.real_data:
        for batch_idx, batch_data in enumerate(real_dataloader):
            for key in batch_data:
                if type(batch_data[key]) is list:
                    for i in range(len(batch_data[key])):
                        batch_data[key][i] = batch_data[key][i].to(device)
                else:
                    batch_data[key] = batch_data[key].to(device)
            # Forward pass
            with torch.no_grad():
                end_points,f_max1 = net(batch_data)
                #数据增强因子
            with torch.no_grad():
                R,scale,noise = aug(batch_data['xyz'][0])
                R_list.append(R)
                scale_list.append(scale)
                noise_list.append(noise)
            for i in range(cfg.num_layers):
                R = R[:, :batch_data['xyz'][i].shape[1] // cfg.sub_sampling_ratio[i], :,:]
                scale = scale[:, :batch_data['xyz'][i].shape[1] // cfg.sub_sampling_ratio[i], :]
                noise = noise[:, :batch_data['xyz'][i].shape[1] // cfg.sub_sampling_ratio[i], :]
                R_list.append(R)
                scale_list.append(scale)
                noise_list.append(noise)
            real_logits = end_points['logits']
            real_logits = real_logits.transpose(1, 2).reshape(-1, cfg.num_classes)
            break

    for batch_idx, batch_data in enumerate(training_dataloader):
        t_start = time.time()
        for key in batch_data:
            if type(batch_data[key]) is list:
                for i in range(len(batch_data[key])):
                    batch_data[key][i] = batch_data[key][i].to(device)
            else:
                batch_data[key] = batch_data[key].to(device)
        #数据增强
        if FLAGS.real_data:
            random_number = (torch.rand(1) * 0.1 + 0.95).to(device)
            # print(batch_idx)
            for i, xyz_layer in enumerate(batch_data['xyz']):
                # 将变量移动到 GPU 上
                R_on_device = R_list[i].to(device)  * random_number
                scale_on_device = scale_list[i].to(device) *  random_number
                noise_on_device = noise_list[i].to(device)  * random_number
                # 矩阵乘法和其他操作
                transformed_xyz = torch.matmul(torch.unsqueeze(xyz_layer, dim=-2), R_on_device).squeeze(-2)
                transformed_xyz = transformed_xyz * scale_on_device
                batch_data['xyz'][i] = transformed_xyz + noise_on_device

        # Forward pass
        optimizer.zero_grad()
        end_points,f_max2 = net(batch_data)
        if FLAGS.real_data:
            loss, end_points = compute_loss1(end_points, cfg, device,real_logits,f_max1,f_max2)

        else:
            loss, end_points = compute_loss(end_points, cfg, device)
        loss.backward()
        optimizer.step()
        acc, end_points = compute_acc(end_points)
        iou_calc.add_data(end_points)               # 保存训练结果，用于计算iou
    
        # Accumulate statistics and print out           # 累计损失和准确率
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'iou' in key:
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()

        batch_interval = 50                             # 本来是10
        if (batch_idx + 1) % batch_interval == 0:
            t_end = time.time()       
            train_loss += stat_dict['loss'] / batch_interval
            train_accuracy += stat_dict['acc'] / batch_interval
            log_string('Step %03d Loss %.3f Acc %.2f lr %.5f --- %.2f ms/batch' % (batch_idx + 1, stat_dict['loss'] / batch_interval, stat_dict['acc'] / batch_interval, optimizer.param_groups[0]['lr'], 1000 * (t_end - t_start)))
            stat_dict['loss'], stat_dict['acc'] = 0, 0
    mean_iou, iou_list = iou_calc.compute_iou()
    log_string('mean IoU:{:.1f}'.format(mean_iou * 100))
    s = 'IoU:'
    for iou_tmp in iou_list:
        s += '{:5.2f} '.format(100 * iou_tmp)
    log_string(s)
    return (train_loss/10),(train_accuracy/10)

def evaluate_one_epoch():
    stat_dict = {} # collect statistics
    net.eval() # set model to eval mode (for bn and dp)
    iou_calc = IoUCalculator(cfg)
    eval_loss = 0
    eval_accuracy = 0
    for batch_idx, batch_data in enumerate(validation_dataloader):
        for key in batch_data:
            if type(batch_data[key]) is list:
                for i in range(len(batch_data[key])):
                    batch_data[key][i] = batch_data[key][i].to(device)
            else:
                batch_data[key] = batch_data[key].to(device)

        # Forward pass
        with torch.no_grad():
            end_points,f_max = net(batch_data)
        if FLAGS.real_data:
            real_logits = end_points['logits']
            real_logits = real_logits.transpose(1, 2).reshape(-1, cfg.num_classes)
            loss, end_points = compute_loss1(end_points, cfg, device,real_logits,f_max,f_max)
        else:
            loss, end_points = compute_loss(end_points, cfg, device)
        acc, end_points = compute_acc(end_points)
        iou_calc.add_data(end_points)

        # Accumulate statistics and print out
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'iou' in key:       # 没有iou一项，iou在下面计算
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()
        eval_loss += stat_dict['loss']
        eval_accuracy += stat_dict['acc']
    for key in sorted(stat_dict.keys()):
        log_string('eval mean %s: %f'%(key, stat_dict[key]/(float(batch_idx+1))))
    mean_iou, iou_list = iou_calc.compute_iou()
    log_string('mean IoU:{:.1f}%'.format(mean_iou * 100))
    log_string('--------------------------------------------------------------------------------------')
    s = f'{mean_iou*100:.1f} | '
    for iou_tmp in iou_list:
        s += '{:5.2f} '.format(100 * iou_tmp)
    log_string(s)
    log_string('--------------------------------------------------------------------------------------')
    return mean_iou,(eval_loss/(float(batch_idx+1))),(eval_accuracy/(float(batch_idx+1)))


def train(start_epoch):
    global EPOCH_CNT
    loss = 0
    now_miou = 0
    max_miou = 0
    logdir='logs'
    writer = SummaryWriter(log_dir=logdir)
    for epoch in range(start_epoch, FLAGS.max_epoch):
        EPOCH_CNT = epoch
        log_string('**** EPOCH %03d ****' % (epoch))
        log_string(str(datetime.now()))
        #train_one_epoch()
        train_loss,train_accuracy= train_one_epoch()
        writer.add_scalar('train_loss',train_loss,epoch+1)
        writer.add_scalar('train_accuracy',train_accuracy,epoch+1)
        #if EPOCH_CNT == 0 or EPOCH_CNT % 10 == 9: # Eval every 10 epochs
        log_string('**** EVAL EPOCH %03d START****' % (epoch))
        now_miou,eval_loss,eval_accuracy = evaluate_one_epoch()
        writer.add_scalar('eval_loss',eval_loss,epoch+1)
        writer.add_scalar('eval_accuracy',eval_accuracy,epoch+1)
        # Save checkpoint
        if(now_miou>max_miou):       # 保存最好的iou的模型
            save_dict = {'epoch': epoch+1, # after training one epoch, the start_epoch should be epoch+1
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                        }
            try: # with nn.DataParallel() the net is added as a submodule of DataParallel
                save_dict['model_state_dict'] = net.module.state_dict()
            except:
                save_dict['model_state_dict'] = net.state_dict()
            torch.save(save_dict, os.path.join(LOG_DIR, 'checkpoint.tar'))
            max_miou = now_miou

        log_string('Best mIoU = {:2.2f}%'.format(max_miou*100))
        log_string('**** EVAL EPOCH %03d END****' % (epoch))
        log_string('')
    writer.close()

if __name__ == '__main__':

    train(start_epoch)