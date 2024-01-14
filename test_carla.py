import torch
import numpy as np
import os
import torch.optim as optim
from os.path import exists, join
from sklearn.metrics import confusion_matrix
import time
from Network import Network, compute_loss, compute_acc, IoUCalculator
from carla_dataset import carla, carlaSampler
from torch.utils.data import DataLoader
from helper_tool import Configcarla as cfg
import torch.nn.functional as F
import os, argparse
import random
from ply import read_ply as read
from helper_ply import write_ply
seed =3407
torch.manual_seed(seed) # 为CPU设置随机种子
torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.	
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
parser = argparse.ArgumentParser()

parser.add_argument('--checkpoint_path', default='UDA/train_output_carla/2024-01-10_13-51-12/checkpoint.tar', help='Model checkpoint path [default: None]')
parser.add_argument('--gpu', type=int, default=0, help='which gpu do you want to use [default: 2], -1 for cpu')
FLAGS = parser.parse_args()

file ='UDA/data/carla/original_ply'

dataset = carla()
test_dataset = carlaSampler(dataset, 'validation')
test_dataloader = DataLoader(test_dataset, batch_size=cfg.val_batch_size, shuffle=True, collate_fn=test_dataset.collate_fn,num_workers=2)

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
net.to(device)
optimizer = optim.Adam(net.parameters(), lr=cfg.learning_rate)
checkpoint_path = FLAGS.checkpoint_path
print(os.path.isfile(checkpoint_path))
if checkpoint_path is not None and os.path.isfile(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("Model restored from %s" % checkpoint_path)
else:
   raise ValueError('CheckPointPathError')
def log_string(out_str, log_out):
    log_out.write(out_str + '\n')
    log_out.flush()
    print(out_str)

class ModelTester:
    def __init__(self,dataset, config ):
        # Add a softmax operation for predictions
        self.softmax = torch.nn.Softmax(dim=1)
        self.test_probs = [np.zeros(shape=[l.shape[0], cfg.num_classes], dtype=np.float32) # 初始化一个全零矩阵,用于放入场景所有点的预测
                    for l in dataset.input_labels['validation']]
        self.config = config
        self.log_out = open('log_test_' + dataset.name + '.txt', 'a')

    def test(self, dataset, num_votes=100, eval=False):
        # Smoothing parameter for votes
        test_smooth = 0.95
        net.eval()
        # Test saving path
        saving_path = time.strftime('results/%Y-%m-%d_%H-%M-%S', time.gmtime())
        test_path = join('/home/vge/UDA/test_output_carla', saving_path.split('/')[-1])
        os.makedirs(test_path) if not exists(test_path) else None
        os.makedirs(join(test_path, 'predictions')) if not exists(join(test_path, 'predictions')) else None
        log_file_name = f'log_test_{dataset.val_split}.txt'
        self.log_out = open(os.path.join(test_path, log_file_name), 'a')      # 追加写入模式
        #####################
        # Network predictions
        #####################
        step_id = 0
        epoch_id = 0
        last_min = -0.5
        t0 = time.time()

        while last_min < num_votes:
            try:
                for batch_idx, batch_data in enumerate(test_dataloader):
                    # # Get data from the dataset

                    for key in batch_data:
                        if type(batch_data[key]) is list:
                            for i in range(len(batch_data[key])):
                                batch_data[key][i] = batch_data[key][i].to(device)
                        else:
                            batch_data[key] = batch_data[key].to(device)

                    # Forward pass
                    with torch.no_grad():
                        end_points,_ = net(batch_data)

                    stacked_probs = end_points['logits']         #测试时不忽略标签
                    stacked_labels = end_points['labels']          #直接从batch中提取数据

                    stacked_probs = stacked_probs.transpose(1, 2).reshape(-1, cfg.num_classes)        # 将logit和label的batch维度消去数据下放到点数目的维度
                    stacked_labels = stacked_labels.reshape(-1)

                    point_idx = end_points['input_inds'].cpu().numpy()
                    cloud_idx = end_points['cloud_inds'].cpu().numpy()
                    correct = torch.sum(torch.argmax(stacked_probs, axis=1) == stacked_labels)        # 计算准确预测的点数
                    acc = (correct / float(np.prod(stacked_labels.shape))).cpu().numpy()             # 计算正确率
                    log_string('step' + str(step_id) + ' acc:' + str(acc), self.log_out)
                    stacked_probs = torch.reshape(stacked_probs, [cfg.val_batch_size, cfg.num_points,
                                                cfg.num_classes])
                    stacked_probs = F.softmax(stacked_probs, dim=2).cpu().numpy()
                    stacked_labels = stacked_labels.cpu().numpy()


                    for j in range(np.shape(stacked_probs)[0]):     # batchsize次（20次）循环，这个for没看懂，看懂了就知道每个场景下的点云正确率怎么来的了
                        probs = stacked_probs[j, :, :]      # 取这个batch下第j个的预测结果（分数）
                        p_idx = point_idx[j, :]             # 取这个batch下第j个的场景中，本次预测结果的点的序号
                        c_i = cloud_idx[j][0]               # 预测的结果来自第j个场景，c_i是该点云的编号
                        self.test_probs[c_i][p_idx] = test_smooth * self.test_probs[c_i][p_idx] + (1 - test_smooth) * probs # 相当于是一个互补滤波？ 对预测分数进行更新（累加）这里应该就是vote的核
                    step_id +=1
                # Save predicted cloud
            #     #new_min = np.min(dataset.min_possibility['validation'])
                new_min = 7.7
                log_string('Epoch {:3d}, end. Min possibility = {:.1f}'.format(epoch_id, new_min), self.log_out)

                if last_min + 1 < new_min:

                    print('Prediction done in {:.1f} s\n'.format(time.time() - t0))
                    print('Saving clouds')

                    # Update last_min
                    last_min = new_min

                    # Project predictions
                    print('\nReproject Vote #{:d}'.format(int(np.floor(new_min))))
                    t1 = time.time()
                    i_test = 0

                    for i, file_path in enumerate(dataset.val_files):

                        # Get file
                        points, gt = self.load_evaluation_points(file_path)
                        gt = gt
                        # Reproject probs
                        probs = np.zeros(shape=[np.shape(points)[0], cfg.num_classes], dtype=np.float16)
                        proj_index = dataset.val_proj[i_test]

                        probs = self.test_probs[i_test][proj_index, :]
                        # Get the predicted labels
                        preds = dataset.label_values[np.argmax(probs, axis=1)].astype(np.uint8)
                        self.evaluate(self,preds, gt)
                        #--------------label select-------------
                        # ####################################################
                        mask =(np.max(probs,axis=1)- np.mean(probs,axis=1))* 10 > 2
                        save_points = points[mask]
                        save_labels = preds[mask]
                        gt_labels = gt[mask]
                        # Save plys
                        cloud_name = file_path.split('/')[-1]
                        ply_name = join(test_path, 'predictions', cloud_name)
                        # Save points with labels for confidence_scores > 0.6
                        write_ply(ply_name, (save_points, save_labels), ['x', 'y', 'z', 'class'])
                        # Save points without labels for confidence_scores <= 0.6
                        no_label_mask = ~mask
                        no_label_points = points[no_label_mask]
                        write_ply(ply_name.replace('.ply', '_no_label.ply'), (no_label_points,), ['x', 'y', 'z'])
                        self.evaluate(self,save_labels, gt_labels)
                        # ######################################################

                        # # Save plys
                        cloud_name = file_path.split('/')[-1] + '_pre'
                        ply_name = join(test_path, 'predictions', cloud_name)
                        write_ply(ply_name, (points, preds), ['x', 'y', 'z', 'class'])
                        log_string(ply_name + ' has been saved', self.log_out)

                        i_test += 1

                    t2 = time.time()
                    print('Reprojection and saving done in {:.1f} s\n'.format(t2 - t1))
                    return

                epoch_id += 1

            except StopIteration:
                break

    
    @staticmethod
    def load_evaluation_points(file_path):
        ply_data = read(file_path)
        points = np.vstack((ply_data['x'], ply_data['y'], ply_data['z'])).T
        labels = ply_data['class'].astype(np.uint8)
        return points, labels

    @staticmethod
    def evaluate(self, pred, gt):
        gt_classes = [0 for _ in range(self.config.num_classes)]
        positive_classes = [0 for _ in range(self.config.num_classes)]
        true_positive_classes = [0 for _ in range(self.config.num_classes)]
        val_total_correct = 0
        val_total_seen = 0

        # Convert numpy arrays to PyTorch tensors
        pred = torch.from_numpy(pred)
        gt = torch.from_numpy(gt)

        if not cfg.ignored_label_inds:
            pred_valid = pred
            labels_valid = gt
        else:
            valid_indices = torch.nonzero(gt != -1).squeeze()
            # 获取未被忽略的标签和预测值
            labels_valid = torch.index_select(gt, 0, valid_indices)
            pred_valid = torch.index_select(pred, 0, valid_indices)
        correct = torch.sum(pred_valid == labels_valid)
        val_total_correct += correct.item()
        val_total_seen += len(labels_valid)

        # Convert PyTorch tensors to numpy arrays
        labels_valid = labels_valid.cpu().numpy()
        pred_valid = pred_valid.cpu().numpy()

        conf_matrix = confusion_matrix(labels_valid, pred_valid, labels=np.arange(0, self.config.num_classes, 1))
        gt_classes += np.sum(conf_matrix, axis=1)
        positive_classes += np.sum(conf_matrix, axis=0)
        true_positive_classes += np.diagonal(conf_matrix)

        iou_list = []
        for n in range(0, self.config.num_classes, 1):
            iou = true_positive_classes[n] / float(gt_classes[n] + positive_classes[n] - true_positive_classes[n])
            iou_list.append(iou)
        mean_iou = sum(iou_list) / float(self.config.num_classes)

        log_string('eval accuracy: {}'.format(val_total_correct / float(val_total_seen)), self.log_out)
        log_string('mean IOU: {}'.format(mean_iou), self.log_out)

        mean_iou = 100 * mean_iou
        log_string('Mean IoU = {:.1f}%'.format(mean_iou), self.log_out)
        s = '{:5.2f} | '.format(mean_iou)
        for IoU in iou_list:
            s += '{:5.2f} '.format(100 * IoU)
        log_string('-' * len(s), self.log_out)
        log_string(s, self.log_out)
        log_string('-' * len(s) + '\n', self.log_out)
        return mean_iou

if __name__ == '__main__':
    test_model = ModelTester(dataset,cfg)
    test_model.test(dataset)