#!/usr/bin/python

import os
import torch 

from tqdm import tqdm 

import pdb 

saved_file_pre = 'wei-indicator-acc_dict_ep-' 
all_epoch = 359 
step = 1 

#gather_p_handle = open('gather_p_val.txt', 'a') 
#gather_p0_handle = open('gather_p0_val.txt', 'a') 
#gather_ind_handle = open('gather_ind_val.txt', 'a') 
#s10gather_p_handle = open('s10_gather_p_val.txt', 'a') 
#s10gather_p0_handle = open('s10_gather_p0_val.txt', 'a') 
#s10gather_ind_handle = open('s10_gather_ind_val.txt', 'a') 
#s100gather_p_handle = open('s100_gather_p_val.txt', 'a') 
#s100gather_p0_handle = open('s100_gather_p0_val.txt', 'a') 
#s100gather_ind_handle = open('s100_gather_ind_val.txt', 'a') 
#s1000gather_p_handle = open('s1000_gather_p_val.txt', 'a') 
#s1000gather_p0_handle = open('s1000_gather_p0_val.txt', 'a') 
#s1000gather_ind_handle = open('s1000_gather_ind_val.txt', 'a') 
#acc_ind_handle = open('gather_acc_val.txt', 'a')
#trainacc_ind_handle = open('gather_acc_trainval.txt', 'a')

gather_mp_handle = open('gather_p_val.txt', 'a') 
gather_mp0_handle = open('gather_p0_val.txt', 'a') 
gather_mp_s1_handle = open('gather_p_s1_val.txt', 'a') 
gather_mp0_s1_handle = open('gather_p0_s1_val.txt', 'a') 
gather_mp_s2_handle = open('gather_p_s2_val.txt', 'a') 
gather_mp0_s2_handle = open('gather_p0_s2_val.txt', 'a') 

count_bar = tqdm(total=all_epoch/step) 

for num in range(1, all_epoch, step): 

    count_bar.update(1) 

    p_val_list = [] 
    p0_val_list = []
    acc_list = [] 
    ind_list = [] 
    cnt_list = [] 
    pred_list = [] 
    label_list = [] 

    file_name = saved_file_pre + str(num) + '.pt' 
    data = torch.load(file_name) 

    for name, raw_sub_data in data.items(): 
        if name == 'val_acc': 
            #acc_ind_handle.write(str(raw_sub_data[0])+ '\n') 
            continue 
        if name == 'train_acc': 
            #trainacc_ind_handle.write(str(raw_sub_data[0])+ '\n')
            continue 

        sub_data = raw_sub_data[0]
        p_val_list.append(sub_data['p_val']) 
        p0_val_list.append(sub_data['p0_val']) 
        ind_list.append(sub_data['ind_val']) 
        cnt_list.append(sub_data['save_cnt']) 
        pred_list.append(sub_data['pred_pos']) 
        label_list.append(sub_data['label']) 

    all_num = len(cnt_list) 

    p_mean = sum(p_val_list) / len(p_val_list) 
    p0_mean = sum(p0_val_list) / len(p_val_list) 
    gather_mp_handle.write(str(p_mean)+ '\n') 
    gather_mp0_handle.write(str(p0_mean)+ '\n') 
    if num % 2 == 0: 
        gather_mp_s2_handle.write(str(p_mean)+ '\n')
        gather_mp0_s2_handle.write(str(p0_mean)+ '\n') 
    else: 
        gather_mp_s1_handle.write(str(p_mean)+ '\n')
        gather_mp0_s1_handle.write(str(p0_mean)+ '\n') 



    #if num > 160: 
    #    break 

    #for cnt in cnt_list: 
    #    cnt_index = cnt_list.index(cnt) 
    #    gather_p_handle.write(str(p_val_list[cnt_index])+ '\n') 
    #    gather_p0_handle.write(str(p0_val_list[cnt_index])+ '\n') 
    #    gather_ind_handle.write(str(ind_list[cnt_index])+ '\n') 
    #
    #    if cnt %10 == 0: 
    #        s10gather_p_handle.write(str(p_val_list[cnt_index])+ '\n') 
    #        s10gather_p0_handle.write(str(p0_val_list[cnt_index])+ '\n') 
    #        s10gather_ind_handle.write(str(ind_list[cnt_index])+ '\n') 

    #    if cnt %100 == 0: 
    #        s100gather_p_handle.write(str(p_val_list[cnt_index])+ '\n') 
    #        s100gather_p0_handle.write(str(p0_val_list[cnt_index])+ '\n') 
    #        s100gather_ind_handle.write(str(ind_list[cnt_index])+ '\n') 

    #    if cnt %1000 == 0: 
    #        s1000gather_p_handle.write(str(p_val_list[cnt_index])+ '\n') 
    #        s1000gather_p0_handle.write(str(p0_val_list[cnt_index])+ '\n') 
    #        s1000gather_ind_handle.write(str(ind_list[cnt_index])+ '\n') 
count_bar.close() 

gather_mp_handle.close() 
gather_mp0_handle.close()  
gather_mp_s1_handle.close()   
gather_mp0_s1_handle.close() 
gather_mp_s2_handle.close()  
gather_mp0_s2_handle.close() 
#gather_p_handle.close() 
#gather_p0_handle.close()
#gather_ind_handle.close() 
#s10gather_p_handle.close() 
#s10gather_p0_handle.close()
#s10gather_ind_handle.close() 
#s100gather_p_handle.close() 
#s100gather_p0_handle.close()
#s100gather_ind_handle.close() 
#s1000gather_p_handle.close() 
#s1000gather_p0_handle.close()
#s1000gather_ind_handle.close() 
#acc_ind_handle.close() 






