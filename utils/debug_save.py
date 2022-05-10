import os
improt cv2
import numpy as np
import torch

def debug_save(self, cur_names, bank_names, cur_inds, bank_inds, weight_gather):
    data_root = '../dataset/CUB_200_2011/dataset/data/'
    save_folder = './vis_tmp_save'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    for cur_img_name, bank_name, cur_img_ind, bank_ind, wei in zip(cur_names, bank_names, cur_inds, bank_inds, weight_gather):

        cur_raw_img = cv2.imread(os.path.join(data_root, cur_img_name))
        save_name = cur_img_name.split('/')[-1][:-4]
        bank_raw_img = cv2.imread(os.path.join(data_root, bank_name))
        x_step = int(cur_raw_img.shape[1] / 14)
        y_step = int(cur_raw_img.shape[0] / 14)
        bx_step = int(bank_raw_img.shape[1] / 14)
        by_step = int(bank_raw_img.shape[0] / 14)
        cur_piece_gather = []
        bank_piece_gather = []
        counter = 0 
        for ind, bind in zip(cur_img_ind, bank_ind):
            cur_img = cur_raw_img.copy()#cv2.imread(os.path.join(data_root, cur_img_name))
            bank_img = bank_raw_img.copy()#cv2.imread(os.path.join(data_root, bank_name))

            coord_x = (ind // 14).int().item()
            coord_y = (ind % 14).int().item() 
            coord_by = (bind // 14).int().item()
            coord_bx = (bind % 14).int().item() 

            cur_piece = cv2.rectangle(cur_img, (coord_x*x_step, coord_y*y_step), (coord_x*x_step + x_step, coord_y*y_step + y_step), (0, 0, 255), 5)
            bank_piece = cv2.rectangle(bank_img, (coord_bx*bx_step, coord_by*by_step), (coord_bx*bx_step + bx_step, coord_by*by_step + by_step), (0, 0, 255), 5)
            #cur_piece = cur_img[coord_x * x_step : (coord_x + 1) * x_step, coord_y * y_step : (coord_y + 1) * y_step]
            #bank_piece = bank_img[coord_bx * bx_step : (coord_bx + 1) * bx_step, coord_by * by_step : (coord_by + 1) * by_step] 
            canvas = np.zeros((300, 620, 3))
            canvas[:, :300, :] = cv2.resize(cur_piece, (300, 300))
            canvas[:, 320:, :] = cv2.resize(bank_piece, (300, 300))
            cv2.imwrite(os.path.join(save_folder, save_name + '_' + str(counter) + '_unsorted.png'), canvas)
            counter += 1
         
        sort_cor = torch.sort(wei)[1]
        counter = 0 
        re_bank_ind = bank_ind[sort_cor] 
        for ind, bind in zip(cur_img_ind, re_bank_ind):
            cur_img = cur_raw_img.copy()#cv2.imread(os.path.join(data_root, cur_img_name))
            bank_img = bank_raw_img.copy()#cv2.imread(os.path.join(data_root, bank_name))

            coord_y = (ind // 14).int().item()
            coord_x = (ind % 14).int().item() 
            coord_by = (bind // 14).int().item()
            coord_bx = (bind % 14).int().item() 
            #cur_piece = cur_img[coord_x * x_step : (coord_x + 1) * x_step, coord_y * y_step : (coord_y + 1) * y_step]
            #bank_piece = bank_img[coord_bx * bx_step : (coord_bx + 1) * bx_step, coord_by * by_step : (coord_by + 1) * by_step] 
            cur_piece = cv2.rectangle(cur_img, (coord_x*x_step, coord_y*y_step), (coord_x*x_step + x_step, coord_y*y_step + y_step), (0, 0, 255), 5)
            bank_piece = cv2.rectangle(bank_img, (coord_bx*bx_step, coord_by*by_step), (coord_bx*bx_step + bx_step, coord_by*by_step + by_step), (0, 0, 255), 5)
            canvas = np.zeros((300, 620, 3))
            canvas[:, :300, :] = cv2.resize(cur_piece, (300, 300))
            canvas[:, 320:, :] = cv2.resize(bank_piece, (300, 300))
            cv2.imwrite(os.path.join(save_folder, save_name + '_' + str(counter) + '_sorted.png'), canvas)
            counter += 1



