from cramerGan.Models import MLP_D as Disc
from cramerGan.Models import MLP_G as Gen

from cramerGan.cramerGan import train_gans
from data import mnist
import numpy as np
import argparse
import torch, os

class DataIterator:
    def __init__(self, train_gen):
        '''
        data: a hdf5 opened pointer
        Index: one array index, [3,5,1,9, 100] index of candidates 
        '''
        self.__dict__.update(locals())

    def next(self):
        return self.__next__()
    def __next__(self):
        images,targets = self.train_gen().__next__()
        return images.reshape( (images.shape[0], 1, 28, 28))

    def __iter__(self):
        return self


if  __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Gans')    
    parser.add_argument('--weight_decay', type=float, default= 0,
                        help='weight decay for training')
    parser.add_argument('--maxepoch', type=int, default=12800000, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--g_lr', type=float, default = 0.00005, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--d_lr', type=float, default = 0.00005, metavar='LR',
                        help='learning rate (default: 0.01)')
    
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--reuse_weigths', action='store_false', default = False,
                        help='continue from last checkout point')
    parser.add_argument('--show_progress', action='store_false', default = True,
                        help='show the training process using images')

    parser.add_argument('--cuda', action='store_false', default=True,
                        help='enables CUDA training')
    
    parser.add_argument('--save_freq', type=int, default= 200, metavar='N',
                        help='how frequent to save the model')
    parser.add_argument('--display_freq', type=int, default= 100, metavar='N',
                        help='plot the results every {} batches')
    
    parser.add_argument('--batch_size', type=int, default= 16, metavar='N',
                        help='batch size.')

    parser.add_argument('--gp_lambda', type=int, default=10, metavar='N',
                        help='the channel of each image.')
    
    parser.add_argument('--noise_dim', type=int, default=10, metavar='N',
                        help='dimension of gaussian noise.')
    parser.add_argument('--ncritic', type=int, default= 5, metavar='N',
                        help='the channel of each image.')
    parser.add_argument('--save_folder', type=str, default= 'tmp_images', metavar='N',
                        help='folder to save the temper images.')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    args.cuda = args.cuda and torch.cuda.is_available()

    netD = Disc(input_size = 28, num_chan =1, hid_dim = 64, out_dim = 16 )
    netG = Gen(input_size  = 28, noise_dim = args.noise_dim, num_chan=1, hid_dim= 64)

    if args.cuda:
        netD = netD.cuda()
        netG = netG.cuda()

    train_gen, dev_gen, test_gen = mnist.load(args.batch_size)
    data_sampler = DataIterator(train_gen).next
    model_root, model_name = 'model', 'mnist_cramer'
    
    train_gans(data_sampler, model_root, model_name, netG, netD,args)
