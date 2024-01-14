from helper_tool import DataProcessing as DP
from helper_tool import Configcarla as cfg
from os.path import join
import numpy as np
import random
import time, pickle, argparse, glob, os
from os.path import join
from helper_ply import read_ply
from torch.utils.data import DataLoader, Dataset, IterableDataset
import torch

# read the subsampled data and divide the data into training and validation
class carla(Dataset):
    def __init__(self):
        self.name = 'carla'
        self.path = 'UDA/data/carla'
        self.original = self.path + 'original_ply'
        self.label_to_names = {0: 'ground',
                               1: 'building',
                               2: 'pole',
                               3: 'natural',
                               4: 'car'  }  
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])        # 进行升序排序,将列表转换为ndarray格式
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}             # {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12}
        self.ignored_labels = np.array([])                                              # 这个数据集上没有ignored标签

        cfg.ignored_label_inds = [self.label_to_idx[ign_label] for ign_label in self.ignored_labels]
        cfg.class_weights = DP.get_class_weights('carla')
        cfg.name = 'carla'

        self.set = set

        # carla
        self.cloud_names = ['Town01','Town02', 'Town03', 'Town05']
        # real data
        self.real_data = ['Lille2']
        self.val_split = self.cloud_names[0]
        self.all_files = glob.glob(join(self.path, 'original_ply', '*.ply'))        # 获取所有的ply文件，返回一个列表   
        self.val_files = glob.glob(join(self.path, 'original_ply', '{}.ply'.format(self.val_split)))


        self.size = len(self.all_files)  

        # Initiate containers
        self.val_proj = []
        self.val_labels = []
        self.possibility = {}
        self.min_possibility = {}
        self.input_trees = {'training': [], 'real': [],'validation': []}
        self.input_colors = {'training': [], 'real': [],'validation': []}
        self.input_labels = {'training': [], 'real': [],'validation': []}
        self.input_names = {'training': [], 'real': [],'validation': []}
        self.load_sub_sampled_clouds(cfg.sub_grid_size)

        print('Size of training : ', len(self.input_colors['training']))                # 训练集有多少个场景（112）（Area2-4）
        print('Size of validation : ', len(self.input_colors['validation']))            # 验证集有44个场景（Area1）
        print('Size of real : ', len(self.input_colors['real']))  
    def load_sub_sampled_clouds(self, sub_grid_size):
        tree_path = join(self.path, 'input_{:.3f}'.format(sub_grid_size))
        for i, file_path in enumerate(self.all_files):
            t0 = time.time()
            cloud_name = file_path.split('/')[-1][:-4]
            #if self.val_split in cloud_name:                # 云名字(字符串)中是否有指定的区域名字（子字符串）
            if cloud_name in self.val_split: 
                cloud_split = 'validation'
            elif cloud_name in self.real_data:
                cloud_split = 'real'
            else:
                cloud_split = 'training'

            # Name of the input files
            kd_tree_file = join(tree_path, '{:s}_KDTree.pkl'.format(cloud_name))            # 读的是采样后的数据
            sub_ply_file = join(tree_path, '{:s}.ply'.format(cloud_name))

            data = read_ply(sub_ply_file)                                                   # data['red'] 就这么读出来的是一个一维向量，存放了所有red的颜色深度
            #sub_colors = np.vstack((data['reflectance']))          # 得到一个n*3的矩阵  
            sub_colors = np.vstack((data['x'],data['y'],data['z'])).T       
            sub_labels = data['class']

            # Read pkl with search tree
            with open(kd_tree_file, 'rb') as f:
                search_tree = pickle.load(f)

            self.input_trees[cloud_split] += [search_tree]              # 列表加列表 表示 列表的拼接，input_trees字典中保存了两个列表，每个列表中的元素都是kdtree对象
            self.input_colors[cloud_split] += [sub_colors]
            self.input_labels[cloud_split] += [sub_labels]
            self.input_names[cloud_split] += [cloud_name]

            size = sub_colors.shape[0] * 4 * 7
            print('{:s} {:.1f} MB loaded in {:.1f}s'.format(kd_tree_file.split('/')[-1], size * 1e-6, time.time() - t0))

        print('\nPreparing reprojected indices for testing')

        # Get validation and test reprojected indices       # 用于预测的时候投影回原来大小的点
        for i, file_path in enumerate(self.all_files):      
            t0 = time.time()
            cloud_name = file_path.split('/')[-1][:-4]

            # Validation projection and labels
            if cloud_name in self.val_split: 
                proj_file = join(tree_path, '{:s}_proj.pkl'.format(cloud_name))
                with open(proj_file, 'rb') as f:
                    proj_idx, labels = pickle.load(f)
                self.val_proj += [proj_idx]                 # 子云中离某个原始点云点最近点的索引
                self.val_labels += [labels]
                print('{:s} done in {:.1f}s'.format(cloud_name, time.time() - t0))   



    def __getitem__(self, idx):
        pass

    def __len__(self):
        # Number of clouds 
        return self.size


class carlaSampler(Dataset):

    def __init__(self, dataset, split='training'):
        self.dataset = dataset
        self.split = split
        self.possibility = {}
        self.min_possibility = {}
        self.class_weight = {}

        if split == 'training':
            self.num_per_epoch = cfg.train_steps * cfg.batch_size   
        elif split == 'real':
            self.num_per_epoch = cfg.train_steps * cfg.batch_size
        elif split == 'validation':
            self.num_per_epoch = cfg.val_steps * cfg.val_batch_size

        self.possibility[split] = []
        self.min_possibility[split] = []
        self.class_weight[split] = []
        for i, tree in enumerate(self.dataset.input_colors[split]):
            self.possibility[split] += [np.random.rand(tree.data.shape[0]) * 1e-3]              # 随机生成可能性 为每个场景的每一个点都生成可能性
            self.min_possibility[split] += [float(np.min(self.possibility[split][-1]))]         # 选出每个场景下最小可能性的那个点
        if split == 'training':
            _, num_class_total = np.unique(np.hstack(self.dataset.input_labels[split]), return_counts=True)
            self.class_weight[split] += [np.squeeze([num_class_total / np.sum(num_class_total)], axis=0)]
    def __getitem__(self, item):
        selected_pc, selected_labels, selected_idx, cloud_ind = self.spatially_regular_gen(item, self.split)
        return selected_pc, selected_labels, selected_idx, cloud_ind

    def __len__(self):
        
        return self.num_per_epoch
        # return 2 * cfg.val_batch_size


    def spatially_regular_gen(self, item, split):
        # Choose a random cloud         # 选择可能性最小的那个点所属的场景
        cloud_idx = int(np.argmin(self.min_possibility[split]))  

        # choose the point with the minimum of possibility in the cloud as query point  选择该场景下的最小概率的点作为查询点 point_ind是点的序号
        point_ind = np.argmin(self.possibility[split][cloud_idx])

        # Get all points within the cloud from tree structure   从kdtree中得到这个场景中的所有点的xyz坐标
        points = np.array(self.dataset.input_trees[split][cloud_idx].data, copy=False)

        # Center point of input region  从所有点中选出概率最低的点（索引用上面求得的） center_point形状为(1,3)
        center_point = points[point_ind, :].reshape(1, -1)

        # Add noise to the center point
        noise = np.random.normal(scale=cfg.noise_init / 10, size=center_point.shape)
        pick_point = center_point + noise.astype(center_point.dtype)                    # 添加噪声

        # Check if the number of points in the selected cloud is less than the predefined num_points
        if len(points) < cfg.num_points:    # 最多取40960个点(并不是所有场景都够40960个点，不够的就全部取出来)
            # Query all points within the cloud
            queried_idx = self.dataset.input_trees[split][cloud_idx].query(pick_point, k=len(points))[1][0]
        else:
            # Query the predefined number of points
            queried_idx = self.dataset.input_trees[split][cloud_idx].query(pick_point, k=cfg.num_points)[1][0]

        # Shuffle index
        queried_idx = DP.shuffle_idx(queried_idx)       # 将序号进行重新打乱分配
        # Get corresponding points and colors based on the index
        queried_pc_xyz = points[queried_idx]            # 对xyz信息进行打乱 用列表作为索引，列表里的每个数索引矩阵的行（第一个轴），并按顺序返回，用于打乱矩阵
        # print(split)

        queried_pc_xyz = queried_pc_xyz - pick_point    # 减去中心点，去中心化
        queried_pc_colors = self.dataset.input_colors[split][cloud_idx][queried_idx]
        queried_pc_labels = self.dataset.input_labels[split][cloud_idx][queried_idx]

        # Update the possibility of the selected points
        dists = np.sum(np.square((points[queried_idx] - pick_point).astype(np.float32)), axis=1)    # 计算每个点离中心点的距离
        if split == 'training':
            queried_pc_labels = self.dataset.input_labels[split][cloud_idx][queried_idx]
            queried_pc_labels = np.array([self.dataset.label_to_idx[l] for l in queried_pc_labels])
            queried_pt_weight = np.array([self.class_weight[split][0][n] for n in queried_pc_labels])
        else:
            queried_pt_weight = 1
        delta = np.square(1 - dists / np.max(dists)) * queried_pt_weight   # 这里注意先乘除后加减。 很巧妙地计算更新概率的大小（离中心点越远，要加的概率就越小，越容易在下一次选中心的时候选中）
        self.possibility[split][cloud_idx][queried_idx] += delta    # 这里应该是更新概率，让下一选中心点时不重复
        self.min_possibility[split][cloud_idx] = float(np.min(self.possibility[split][cloud_idx]))  # 更新该场景的最小概率


        # up_sampled with replacement
        if len(points) < cfg.num_points:    # 如果不够40960个点，就使用数据增强到这么多个点
            queried_pc_xyz, queried_pc_colors, queried_idx, queried_pc_labels = \
                DP.data_aug(queried_pc_xyz, queried_pc_colors, queried_pc_labels, queried_idx, cfg.num_points) 


        queried_pc_xyz = torch.from_numpy(queried_pc_xyz).float()           # 转换回张量格式
        queried_pc_colors = torch.from_numpy(queried_pc_colors).float()
        queried_pc_labels = torch.from_numpy(queried_pc_labels).long()
        queried_idx = torch.from_numpy(queried_idx).float() # keep float here?
        cloud_idx = torch.from_numpy(np.array([cloud_idx], dtype=np.int32)).float()
        points = queried_pc_xyz
    
        return points, queried_pc_labels, queried_idx, cloud_idx      


    def tf_map(self, batch_xyz, batch_features, batch_label, batch_pc_idx, batch_cloud_idx):    # 进行下采样和KNN的索引记录，为后面网络做准备
        batch_features = np.concatenate([batch_xyz, batch_features], axis=-1)
        # if batch_xyz.shape[0] == 4:
        #     batch_features = self.torch_augment_input(batch_xyz, batch_features)
        #     # print('data aug')
        # else:
        #     batch_features = np.concatenate([batch_xyz, batch_features], axis=-1)
        #     # print('no')
        input_points = []
        input_neighbors = []
        input_pools = []
        input_up_samples = []
        #neigh_1 = DP.knn_search(batch_xyz, batch_xyz, cfg.k_n) 
        for i in range(cfg.num_layers):     # 每一层的降采样在这里实现（从这里开始不可以再随意打乱矩阵的顺序了，因为knn search依靠的是矩阵的索引找到近邻点）
            neighbour_idx = DP.knn_search(batch_xyz, batch_xyz, cfg.k_n)      # KNN搜索每个点周围16个点，记录点的索引，维度是（6，40960，16）
            sub_points = batch_xyz[:, :batch_xyz.shape[1] // cfg.sub_sampling_ratio[i], :]      # 随机下采样 维度是（6，40690//4，3）
            pool_i = neighbour_idx[:, :batch_xyz.shape[1] // cfg.sub_sampling_ratio[i], :]      # 对索引也随机下采样 （6，40960//4，16）
            up_i = DP.knn_search(sub_points, batch_xyz, 1)                      # KNN搜索每个原点最近的下采样点 维度是（6，40960，1）
            input_points.append(batch_xyz)
            input_neighbors.append(neighbour_idx)
            input_pools.append(pool_i)
            input_up_samples.append(up_i)
            batch_xyz = sub_points


        input_list = input_points + input_neighbors + input_pools + input_up_samples
        input_list += [batch_features, batch_label, batch_pc_idx, batch_cloud_idx]

        return input_list
    #data aug
    def torch_augment_input(self, batch_xyz, batch_features):
        #生成随机角度
        batch_size = batch_xyz.shape[0]
        num_points = batch_xyz.shape[1]
        
        theta = np.random.rand(batch_size, num_points) * (2 * np.pi)

        c, s = np.cos(theta), np.sin(theta)
        cs0 = np.zeros_like(c)
        cs1 = np.ones_like(c)
        R = np.stack([c, -s, cs0, s, c, cs0, cs0, cs0, cs1], axis=-1).reshape(batch_size, num_points, 3, 3)

        # Batch matrix multiplication
        transformed_xyz = np.matmul(np.expand_dims(batch_xyz, axis=-2), R).squeeze(-2)

        min_s = cfg.augment_scale_min
        max_s = cfg.augment_scale_max
        if cfg.augment_scale_anisotropic:
            s = np.random.rand(batch_size, 1, 3) * (max_s - min_s) + min_s
        else:
            s = np.random.rand(batch_size, 1, 1) * (max_s - min_s) + min_s
        
        symmetries = []
        for i in range(3):
            if cfg.augment_symmetries[i]:
                symmetries.append(np.round(np.random.rand(batch_size, 1, 1)) * 2 - 1)
            else:
                symmetries.append(np.ones((batch_size, 1, 1), dtype=np.float32))
        symmetries = np.concatenate(symmetries, axis=2)
        s *= symmetries
        
        stacked_scales = np.repeat(s, num_points, axis=1)
        #print(stacked_scales)
        transformed_xyz = transformed_xyz * stacked_scales
        
        noise = np.random.normal(size=transformed_xyz.shape) * cfg.augment_noise
        # print(noise)
        transformed_xyz = transformed_xyz + noise
        stacked_features = np.concatenate([transformed_xyz, batch_features], axis=-1)
        return stacked_features
    
    # 这个函数是每从dataloader拿一次数据执行一次
    def collate_fn(self,batch):

        selected_pc, selected_labels, selected_idx, cloud_ind = [],[],[],[]
        for i in range(len(batch)):
            selected_pc.append(batch[i][0])
            selected_labels.append(batch[i][1])
            selected_idx.append(batch[i][2])
            cloud_ind.append(batch[i][3])

        selected_pc = np.stack(selected_pc)                     # 将列表堆叠起来形成矩阵，维度为（batch，nums，feature）=（6，40960，6）
        selected_labels = np.stack(selected_labels)
        selected_idx = np.stack(selected_idx)
        cloud_ind = np.stack(cloud_ind)

        selected_xyz = selected_pc[:, :, 0:3]
        selected_features = selected_pc[:, :, 3:6]

        flat_inputs = self.tf_map(selected_xyz, selected_features, selected_labels, selected_idx, cloud_ind) # 返回值是一个包含24个列表的列表

        num_layers = cfg.num_layers
        inputs = {}
        inputs['xyz'] = []
        for tmp in flat_inputs[:num_layers]:
            inputs['xyz'].append(torch.from_numpy(tmp).float())     # 添加了五个列表，每次随机采样前的坐标
        inputs['neigh_idx'] = []
        for tmp in flat_inputs[num_layers: 2 * num_layers]:
            inputs['neigh_idx'].append(torch.from_numpy(tmp).long())    # 添加了五个列表，输入点每次随机采样前的16个邻居的坐标（第一个列表没有进行下采样）
        inputs['sub_idx'] = []
        for tmp in flat_inputs[2 * num_layers:3 * num_layers]:
            inputs['sub_idx'].append(torch.from_numpy(tmp).long())      # 添加了五个列表，输入点的每次随机采样后的16个邻居的坐标
        inputs['interp_idx'] = []
        for tmp in flat_inputs[3 * num_layers:4 * num_layers]:
            inputs['interp_idx'].append(torch.from_numpy(tmp).long())   # 添加了五个列表，输入点每次随机采样后每个原点的最近的下采样点
        inputs['features'] = torch.from_numpy(flat_inputs[4 * num_layers]).float()  # 改了一下，为了适应后面linear的维度，不转置了
        inputs['labels'] = torch.from_numpy(flat_inputs[4 * num_layers + 1]).long()
        inputs['input_inds'] = torch.from_numpy(flat_inputs[4 * num_layers + 2]).long()
        inputs['cloud_inds'] = torch.from_numpy(flat_inputs[4 * num_layers + 3]).long()

        return inputs


if __name__ == '__main__':      # use to test
    dataset = carla()
    dataset_train = carlaSampler(dataset, split='training')
    dataloader = DataLoader(dataset_train, batch_size=cfg.batch_size, shuffle=True, drop_last=True, collate_fn=dataset_train.collate_fn)
    # dataloader = DataLoader(dataset_train, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    for i,data in enumerate(dataloader):
        features = data['features']  #feature是xyz经过多层感知机学习的
        labels = data['labels']
        idx = data['input_inds']
        cloud_idx = data['cloud_inds']
        xyz = data['xyz'] #xyz 包括了每一层下采样的坐标，不是一个单独的xyz
        # print(xyz[0]+1) #只增强了第一层的xyz，第二层恢复来的样子，应该将增强因子应用于每一层，for循环
        print(data['sub_idx'][0].shape)
        print(data['sub_idx'][1].shape)
        print(data['sub_idx'][2].shape)
        print(data['sub_idx'][3].shape)
        print(data['sub_idx'][4].shape)
        break


