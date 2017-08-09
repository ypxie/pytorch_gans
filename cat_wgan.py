# -*- coding: utf-8 -*-

from Gans.Models import DCGAN_D as Disc
from Gans.Models import DCGAN_G as Gen

from Gans.gpGan import train_gans
import numpy as np
import argparse, os
import torch, h5py

home = os.path.expanduser('~')

h5dataFile = os.path.join(home, 'Dataset/cat/CatImg_size_64.h5')

class DataIterator:
    def __init__(self, h5data, batch_size):
        '''
        data: a hdf5 opened pointer
        Index: one array index, [3,5,1,9, 100] index of candidates 
        '''
        self.__dict__.update(locals())
        self.images = h5data['images']
        self.total_imgs = self.images.shape[0]
        self.batch = np.zeros((self.batch_size, )+ self.images.shape[1::]).astype(np.float32)
    def next(self):
        return self.__next__()

    def __next__(self):
        random_indexes = np.random.choice(self.total_imgs,  size=self.batch_size, replace=False)
        for idx, index in enumerate(random_indexes):
                self.batch[idx] = self.images[index]
        return self.batch

    def __iter__(self):
        return self


if  __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Gans')    
    parser.add_argument('--weight_decay', type=float, default= 0,
                        help='weight decay for training')
    parser.add_argument('--maxepoch', type=int, default=12800000, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--g_lr', type=float, default = .0001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--d_lr', type=float, default = .0001, metavar='LR',
                        help='learning rate (default: 0.01)')

    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--reuse_weigths', action='store_false', default = True,
                        help='continue from last checkout point')
    parser.add_argument('--show_progress', action='store_false', default = True,
                        help='show the training process using images')

    parser.add_argument('--cuda', action='store_false', default=True,
                        help='enables CUDA training')
    
    parser.add_argument('--save_freq', type=int, default= 500, metavar='N',
                        help='how frequent to save the model')
    parser.add_argument('--display_freq', type=int, default= 100, metavar='N',
                        help='plot the results every {} batches')
    
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='batch size.')

    parser.add_argument('--gp_lambda', type=int, default=10, metavar='N',
                        help='the channel of each image.')
    parser.add_argument('--gpwgan', action='store_false', default=  False,
                        help='enables gradient penalty')
    
    parser.add_argument('--noise_dim', type=int, default= 320, metavar='N',
                        help='dimension of gaussian noise.')
    parser.add_argument('--ncritic', type=int, default= 5, metavar='N',
                        help='the channel of each image.')
    parser.add_argument('--ngen', type=int, default= 1, metavar='N',
                        help='the channel of each image.')
    
    parser.add_argument('--save_folder', type=str, default= 'tmp_images', metavar='N',
                        help='folder to save the temper images.')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = "3"

    args.cuda = args.cuda and torch.cuda.is_available()
    
    netD = Disc(input_size = 64, num_chan = 3, hid_dim = 64, out_dim = 1, bn=True)
    netG = Gen(input_size  = 64, noise_dim = args.noise_dim, num_chan=3, hid_dim= 64)
    
    if args.cuda:
        netD = netD.cuda()
        netG = netG.cuda()
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
    
    print(netG)
    print(netD)
     
    with h5py.File(h5dataFile, 'r') as h5_data:
        data_sampler = DataIterator(h5_data, args.batch_size).next
        model_root, model_name = 'model', 'cat_Wgan'
        
        train_gans(data_sampler, model_root, model_name, netG, netD,args)
