import os
import argparse

import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pdb

def send_mail(img_list):
    print('send email ...')
    msg = MIMEMultipart()
    msg['Subject'] = 'exp_figures'
    msg['From'] = 'liangyzh18@outlook.com'
    msg['To'] = 'liangyzh18@outlook.com'

    text = MIMEText("exp_result")
    msg.attach(text)

    for img in img_list:
        img_data = open(img, 'rb').read()
        image = MIMEImage(img_data, name=os.path.basename(img))
        msg.attach(image)

    s = smtplib.SMTP('smtp.outlook.com', 587)
    s.ehlo()
    s.starttls()
    s.ehlo()
    s.login("liangyzh18@outlook.com", "killar1995L")
    s.sendmail(msg['From'], msg['To'], msg.as_string())
    s.quit()
    print('send email done ...')



def parse_args():
    parser = argparse.ArgumentParser(description='save para')
    parser.add_argument('--dir', dest='dir', default='./grep_log', type=str)
    args = parser.parse_args()
    return args

class save_figures(object):
    def __init__(self, folder):
        self.saved_imgs = []
        self.folder=folder
        self.data_step = 5000

    def __call__(self, data_list, name):
        raw_data_list = [float(x) for x in data_list]
        data_list = []
        for item in raw_data_list:
            if item != 0:
                data_list.append(item) 
        print('len of nonzero: ', len(data_list))
        if len(data_list) == 0:
            print(name + ' is all zero')
            #pdb.set_trace()
        if len(data_list) > self.data_step:
            sampled_step = len(data_list) // self.data_step
            sampled = [sum(data_list[x*sampled_step:(x+1)*sampled_step])/sampled_step for x in range(self.data_step)]
            self.plot_curve(sampled, 'sampled_'+name)
            self.plot_curve(data_list[-5000:], 'latest_'+name)
        else:
            self.plot_curve(data_list, name)

    def plot_curve(self, data_list, name):
        plt.plot(data_list)
        img_path = os.path.join(self.folder, name+'.png')
        plt.title(name)
        plt.savefig(img_path)
        plt.close()
        self.saved_imgs.append(img_path)

    def get_imgs(self):
        return self.saved_imgs


if __name__ == '__main__':
    args = parse_args()
    folder = args.dir
    exp_plot = save_figures(folder)
    
    trainacc_io = open(os.path.join(folder, 'train_acc'))
    trainacc_data = trainacc_io.readlines()
    valacc_io = open(os.path.join(folder, 'val_acc'))
    valacc_data = valacc_io.readlines()
    as_valacc_io = open(os.path.join(folder, 'asval_acc'))
    as_valacc_data = as_valacc_io.readlines()
    
    debug_loss = os.path.join(folder, 'loss')
    if os.path.exists(debug_loss):
        loss_io = open(debug_loss)
        loss_data = loss_io.readlines()
        print('save loss ...')
        loss_num = len(loss_data[0].split(' '))
        print('loss gathering ...')
        loss_gather = [x.split(' ') for x in loss_data]
        for loss_item in range(loss_num):
            print('loss %d' %loss_item)
            exp_plot([x[loss_item] for x in loss_gather], 'loss'+str(loss_item))

    print('save train acc ...')
    trainacc_gather = [x.split(' ') for x in trainacc_data]
    for acc_item in range(3):
        exp_plot([x[acc_item] for x in trainacc_gather], 'trainacc'+str(1+acc_item))

    print('save val acc ...')
    valacc_gather = [x.split(' ') for x in valacc_data]
    for acc_item in range(3):
        exp_plot([x[0+acc_item] for x in valacc_gather], 'valacc'+str(1+acc_item))

    print('sav as val acc ...')
    valacc_gather = [x.split(' ') for x in as_valacc_data]
    for acc_item in range(3):
        exp_plot([x[acc_item] for x in valacc_gather], 'as_valacc'+str(1+acc_item))


    img_list = exp_plot.get_imgs()
    #send_mail(img_list)






