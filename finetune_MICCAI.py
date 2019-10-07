import argparse
import os
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F

from torch.autograd import Variable

from dataloader import MICCAI_listfile as lister
from dataloader import MICCCAI_fileloader as loader

from models import *

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--maxdisp', type=int, default=256,
                    help='maximum disparity')
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--datatype', default='MICCAI',
                    help='data type')
parser.add_argument('--datapath', default='/media/xiran_zhang/Crypto/MICCAI')
# we should modify epochs to get the best performance, experiments needed here
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train')
parser.add_argument('--loadmodel', default='./finetune_MICCAI/finetune_max256_44.tar',
                    help='load model')
parser.add_argument('--savemodel', default='./finetune_MICCAI/',
                    help='save model')
parser.add_argument('--no-cuda',action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = '0'

print(torch.cuda.is_available())
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)

train_left_img, train_right_img, train_left_disp, val_left_img, val_right_img, val_left_disp, \
test_left_img, test_right_img = lister.dataloader(args.datapath)

TrainImgLoader = torch.utils.data.DataLoader(
    loader.myImageLoader(train_left_img, train_right_img, train_left_disp, True),
    batch_size = 1, shuffle = True, num_workers = 8, drop_last = False)

ValImgLoader = torch.utils.data.DataLoader(
    loader.myImageLoader(val_left_img, val_right_img, val_left_disp, False),
    batch_size = 1, shuffle = False, num_workers = 1, drop_last = False)


TestImgLoader = torch.utils.data.DataLoader(
    loader.testImageLoader(test_left_img, test_right_img),
    batch_size=1, shuffle = False, num_workers = 1, drop_last = False)


if args.model == 'stackhourglass':
    model = stackhourglass(args.maxdisp)
elif args.model == 'basic':
    model = basic(args.maxdisp)
else:
    print('no model')


if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()

pause_epoch = 0


if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])
    pause_epoch = state_dict['epoch']
    max_epo_load = state_dict['max_epo']
    max_acc_load = state_dict['max_acc']

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

optimizer = optim.Adam(model.parameters(), lr=0.1, betas=(0.9, 0.999))


def train(imgL, imgR, disp_L):
    model.train()
    imgL = Variable(torch.FloatTensor(imgL))
    imgR = Variable(torch.FloatTensor(imgR))
    disp_L = Variable(torch.FloatTensor(disp_L))

    if args.cuda:
        imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_L.cuda()

    # ---------
    mask = (disp_true > 0)
    mask.detach_()
    # ----

    optimizer.zero_grad()

    if args.model == 'stackhourglass':
        output1, output2, output3 = model(imgL, imgR)
        output1 = torch.squeeze(output1, 1)
        output2 = torch.squeeze(output2, 1)
        output3 = torch.squeeze(output3, 1)
        loss = 0.5 * F.smooth_l1_loss(output1[mask], disp_true[mask], size_average=True) + 0.7 * F.smooth_l1_loss(
            output2[mask], disp_true[mask], size_average=True) + F.smooth_l1_loss(output3[mask], disp_true[mask],
                                                                                  size_average=True)
    elif args.model == 'basic':
        output3 = model(imgL, imgR)
        output3 = torch.squeeze(output3, 1)
        loss = F.smooth_l1_loss(output3[mask], disp_true[mask], size_average=True)

    loss.backward()
    optimizer.step()

    return loss.data.item()

def val(left, imgL_up, imgR_up, imgL_mid, imgR_mid, imgL_bot, imgR_bot, disp_true):
    model.eval()
    imgL_up = Variable(torch.FloatTensor(imgL_up))
    imgR_up = Variable(torch.FloatTensor(imgR_up))
    imgL_mid = Variable(torch.FloatTensor(imgL_mid))
    imgR_mid = Variable(torch.FloatTensor(imgR_mid))
    imgL_bot = Variable(torch.FloatTensor(imgL_bot))
    imgR_bot = Variable(torch.FloatTensor(imgR_bot))

    if args.cuda:
        imgL_up, imgR_up, imgL_mid, imgR_mid, imgL_bot, imgR_bot = \
            imgL_up.cuda(), imgR_up.cuda(), imgL_mid.cuda(), imgR_mid.cuda(), imgL_bot.cuda(), imgR_bot.cuda()

    with torch.no_grad():
        output3_up = model(imgL_up, imgR_up)
        output3_mid = model(imgL_mid, imgR_mid)
        output3_bot = model(imgL_bot, imgR_bot)

    pred_disp_up = output3_up.data.cpu()
    pred_disp_mid = output3_mid.data.cpu()
    pred_disp_bot = output3_bot.data.cpu()

    pred_disp = torch.zeros(1,1024, 1280)
    print(pred_disp_bot.shape)
    pred_disp[0,0:384,:] = pred_disp_up[0,0:384,:]
    pred_disp[0,384:640,:] = pred_disp_mid[0,128:384,:]
    pred_disp[0,640:1024,:] = pred_disp_bot[0,128:512,:]
    # print(pred_disp.shape)
    # plt.imshow(pred_disp[0])
    # plt.show()
    # fs = cv2.FileStorage("./up51.ext", flags=1)
    # disp_mat = pred_disp[0].numpy()
    # print(disp_mat.shape)
    # fs.write(name='disp', val = disp_mat)
    # fs.release()


        # computing 3-px error#
    true_disp = disp_true
    index = np.argwhere(true_disp > 0)
    disp_true[index[0][:], index[1][:], index[2][:]] = np.abs(true_disp[index[0][:], index[1][:], index[2][:]] - pred_disp[index[0][:], index[1][:], index[2][:]])
    correct = (disp_true[index[0][:], index[1][:], index[2][:]] < 3) | (disp_true[index[0][:], index[1][:], index[2][:]] < true_disp[index[0][:], index[1][:], index[2][:]] * 0.05)
    torch.cuda.empty_cache()

    #
    # if (float(torch.sum(correct)) / float(len(index[0]))) < 0.9  or (float(torch.sum(correct)) / float(len(index[0]))) > 0.98:
    #     print(1 - (float(torch.sum(correct)) / float(len(index[0]))))
    #     print(left)
    #     error_vis = disp_true[0].numpy()
    #     plt.imshow(error_vis)
    #     plt.show()

    if len(index[0]) != 0:
        return 1 - (float(torch.sum(correct)) / float(len(index[0])))
    else:
        return 0

def test(imgL, imgR):
    model.eval()
    imgL = Variable(torch.FloatTensor(imgL))
    imgR = Variable(torch.FloatTensor(imgR))
    if args.cuda:
        imgL, imgR = imgL.cuda(), imgR.cuda()

    with torch.no_grad():
        output3 = model(imgL, imgR)

    pred_disp = output3.data.cpu()
    return pred_disp

def adjust_learning_rate(optimizer, epoch):
    if epoch <= 20:
       lr = 0.001
    else:
       lr = 0.0001
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



def main():
    # max_acc = 100
    # max_epo = 100
    if max_acc_load:
        max_acc = max_acc_load
        max_epo = max_epo_load
    else:
        max_acc = 100
        max_epo = 100

    start_full_time = time.time()

    training = 0


    if(training == 1):


        for epoch in range(pause_epoch+1, args.epochs+1):
            print(epoch)

            start_epoch_time = time.time()
            total_train_loss = 0
            total_test_loss = 0
            adjust_learning_rate(optimizer, epoch)

            # # Train
            # mal_train_count = 0
            # for batch_idx, (imgL_crop, imgR_crop, dispL_crop) in enumerate(TrainImgLoader):
            #     start_time = time.time()
            #
            #     loss = train(imgL_crop, imgR_crop, dispL_crop)
            #
            #     print('Iter %d training loss = %.3f, time =  %.2f' %(batch_idx, loss, time.time() - start_time))
            #     if(loss < 1000):
            #         total_train_loss += loss
            #     else:
            #         mal_train_count += 1
            # print('epoch %d totoal training loss = %.3f'  %(epoch, total_train_loss / (len(TrainImgLoader) - mal_train_count)))

            # Test
            mal_test_count = 0
            for batch_idx, (left, imgL_up, imgR_up, imgL_mid, imgR_mid, imgL_bot, imgR_bot, dispL) in enumerate(ValImgLoader):
                val_loss = val(left, imgL_up, imgR_up, imgL_mid, imgR_mid, imgL_bot, imgR_bot, dispL)
                print('Iter %d 3-px error in val = %.3f' %(batch_idx, val_loss * 100))
                if val_loss > 0 and val_loss < 1:
                    total_test_loss += val_loss
                else:
                    mal_test_count += 1
            acc = total_test_loss/(len(ValImgLoader) - mal_test_count)
            print('epoch %d total 3-px error in vcal = %.3f' %(epoch, acc))



            if total_test_loss/len(ValImgLoader) * 100 < max_acc:
                max_acc = total_test_loss/len(ValImgLoader) * 100
                max_epo = epoch

                print('Max epoch %d: total test error = %.3f' % (max_epo, max_acc))

            log = open("./log.txt", "a")
            log.write('epoch: '+ str(epoch))
            log.write('max epoch: ' + str(max_epo))
            log.write(', max acc: ' + str(max_acc) + '\n')
            log.close()

            # Save model
            savefilename = args.savemodel + 'finetune_max256_' + str(epoch) + '.tar'
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'train_loss': total_train_loss/len(TrainImgLoader),
                'test_loss': total_test_loss/len(ValImgLoader)* 100,
                'max_epo': max_epo,
                'max_acc': max_acc,
            }, savefilename)
            print('epoch %d training time: %.2f HR' % (epoch, (time.time() - start_epoch_time) / 3600))

    else:

        for batch_idx, (left, right, imgL_up, imgR_up, imgL_mid, imgR_mid, imgL_bot, imgR_bot) in enumerate(TestImgLoader):
            predicted_up = test(imgL_up, imgR_up)
            predicted_mid = test(imgL_mid, imgR_mid)
            predicted_bot = test(imgL_bot, imgR_bot)

            result_up = predicted_up[0]
            result_mid = predicted_mid[0]
            result_bot = predicted_bot[0]

            result = np.zeros((1024, 1280))
            result[0:384,:] = result_up[0:384,:]
            result[384:640,:] = result_mid[128:384,:]
            result[640:1024,:] = result_bot[128:512,:]


            file_name= left[0].split('/')[-1].split('.')[0] + str(batch_idx) + str(256)

            fig = plt.figure(figsize=(12.8, 10.24))  # Your image (W)idth and (H)eight in inches
            # Stretch image to full figure, removing "grey region"
            plt.subplots_adjust(left=0.05, right=0.9, top=0.95, bottom=0.05)
            im = plt.imshow(result,vmin=0, vmax=256)  # Show the image
            pos = fig.add_axes([0.93, 0.1, 0.02, 0.8])  # Set colorbar position in fig
            fig.colorbar(im, cax=pos)  # Create the colorbar
            plt.savefig(file_name+'.png')


            path_dir = left[0].split('left_finalpass')[0].split('Crypto')[-1]
            save_path = '/media/xiran_zhang/2011_HDD7' + path_dir+ 'disparity/' + file_name + '.ext'
            print(save_path)
            fs = cv2.FileStorage(save_path, flags=1)
            disparity = result
            print(disparity.shape)
            fs.write(name='disp', val = disparity.astype(np.float32))
            fs.release()

    print('full finetune time = %.2f HR' % ((time.time() - start_full_time)/ 3600))
    print('max epo:', max_epo)
    print('max acc:', max_acc)

if __name__ == '__main__':
    main()