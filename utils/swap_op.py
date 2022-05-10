import os
import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import pdb

class SwapMapping():
    def __init__(self, nei_num, size):
        super(SwapMapping, self).__init__()
        self.nei_num = nei_num
        self.map_size = size
        self.position_map = self.position_matrix()
        self.kernel = self.neighbor_kernel()
    

    def position_matrix(self):
        p_m1 = torch.LongTensor([x for x in range(self.map_size[0])])
        p_m1 = p_m1.repeat(self.map_size[0], 1)
        p_m2 = torch.transpose(p_m1, 0, 1)
        return torch.stack([p_m1, p_m2])
        
    def neighbor_kernel(self):
        gather_kernel = []
        msize = int(math.sqrt(self.nei_num))
        for num in range(self.nei_num):
            kernel = torch.zeros(self.nei_num)
            kernel[num] = 1
            gather_kernel.append(kernel.view(msize, msize))
        nei_kernel = torch.stack(gather_kernel)
        return nn.Parameter(data=nei_kernel, requires_grad=False)

    def neighbor_mapping(self):
        neighbor_map1 = F.conv2d(self.position_map[0].unsqueeze(0).unsqueeze(0).float(), self.kernel.unsqueeze(1), padding=1)
        neighbor_map2 = F.conv2d(self.position_map[1].unsqueeze(0).unsqueeze(0).float(), self.kernel.unsqueeze(1), padding=1)
        neighbor_map = torch.stack([neighbor_map1, neighbor_map2]).squeeze(1)
        random_seq = torch.randint(0, self.nei_num, self.map_size)
        random_map = torch.zeros(2, self.map_size[0], self.map_size[1])
        for corx in range(self.map_size[0]):
            for cory in range(self.map_size[1]):
                random_map[:, corx, cory] = neighbor_map[:, random_seq[corx, cory].long(), corx, cory]
        random_1d_map = random_map[0] + random_map[1] * self.map_size[0]
        random_1d_map = random_1d_map.view(-1)

        val = []
        for num in range(self.map_size[0]*self.map_size[1]):
            if (random_1d_map == num).float().sum() > 1:
                val.append(num)
                random_1d_map = (random_1d_map != num).float() * random_1d_map + ((random_1d_map == num).float() * -1)  
            elif num not in random_1d_map:
                val.append(num)
        cnt = 0
        for pos in range(self.map_size[0]*self.map_size[1]):
            if random_1d_map[pos] == -1:
                random_1d_map[pos] = val[cnt]
                cnt += 1
        map_gt = torch.sort(random_1d_map)
        normal_map = torch.FloatTensor(list(range(self.map_size[0]*self.map_size[1])))
        return random_1d_map, map_gt[0], normal_map 
    
    def polar_trans(self, map_1d):
        hor = map_1d // self.map_size[0]
        ver = map_1d % self.map_size[0]
        hor_c = self.map_size[0] // 2
        ver_c = self.map_size[1] // 2
        dis = (hor - hor_c).pow(2) + (ver - ver_c).pow(2)
        delta = 3.14*(ver > ver_c).float() 
        theta = (hor - hor_c).float() / (dis.float().sqrt() + 1e-5)
        return dis.float().sqrt().cuda(), (theta.acos() + delta).cuda()

if __name__ == '__main__':
    swap_mapping = SwapMapping(9, [7, 7]) 
    map_1d, map_gt = swap_mapping.neighbor_mapping()
    import cv2 
    img = cv2.imread('test.jpg')
    img_tensor = torch.tensor(img).transpose(1,2).transpose(0,1)
    img_padding = torch.zeros(3, 420, 504)
    img_padding[:, :416, :500] = img_tensor
    img_piece = img_padding.view(3, 7, 60, 7, 72).transpose(2,3).contiguous().view(3, 49, 60, 72)
    swap_img = img_piece[:, map_1d.long(), :, :]
    trans_img = swap_img[:, map_gt.long(), :, :]
    
    pdb.set_trace()
 
