import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

activation = torch.nn.SELU

class LayerNormal(nn.Module):
    ''' Layer normalization module '''

    def __init__(self, d_hid, eps=1e-9):
        super(LayerNormal, self).__init__()

        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(1, d_hid, 1, 1), requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(1, d_hid, 1, 1), requires_grad=True)

    def forward(self, z):

        mu   = torch.mean(z, 1, keepdim=True)
        sigma =  torch.std(z, dim=1, keepdim=True )
        
        #print(mu.size(), sigma.size(), z.size(),flat_z.size())
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        
        ln_out = ln_out * self.a_2.expand_as(ln_out) + self.b_2.expand_as(ln_out)
        return ln_out
        
class MLP_G(nn.Module):
    def __init__(self, input_size, noise_dim, num_chan, hid_dim, ngpu=1):
        super(MLP_G, self).__init__()
        self.__dict__.update(locals())
        self.ngpu = ngpu
        self.register_buffer('device_id', torch.zeros(1))
        main = nn.Sequential(
            # Z goes into a linear of size: hid_dim
            nn.Linear(noise_dim, hid_dim),
            activation(),
            nn.Linear(hid_dim, hid_dim),
            activation(),
            nn.Linear(hid_dim, hid_dim),
            activation(),
            nn.Linear(hid_dim, hid_dim),
            activation(),
            nn.Linear(hid_dim, num_chan * input_size * input_size),
            nn.Sigmoid()
        )
        self.main = main
        self.num_chan = num_chan
        self.input_size = input_size
        self.noise_dim = noise_dim

    def forward(self, inputs):
        inputs = inputs.view(inputs.size(0), inputs.size(1))
        if isinstance(inputs.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, inputs, range(self.ngpu))
        else:
            output = self.main(inputs)
        return output.view(output.size(0), self.num_chan, self.input_size, self.input_size)


class MLP_D(nn.Module):
    def __init__(self, input_size, num_chan, hid_dim, out_dim=1, ngpu=1):
        super(MLP_D, self).__init__()
        
        self.__dict__.update(locals())
        self.ngpu = ngpu
        self.register_buffer('device_id', torch.zeros(1))
        main = nn.Sequential(
            # Z goes into a linear of size: hid_dim
            nn.Linear(num_chan * input_size * input_size, hid_dim),
            activation(),
            nn.Linear(hid_dim, hid_dim),
            activation(),
            nn.Linear(hid_dim, hid_dim),
            activation(),
            nn.Linear(hid_dim, hid_dim),
            activation(),
            nn.Linear(hid_dim, out_dim),
        )
        self.main = main
        self.num_chan = num_chan
        self.input_size = input_size

    def forward(self, inputs):
        inputs = inputs.view(inputs.size(0), -1)
        if isinstance(inputs.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, inputs, range(self.ngpu))
        else:
            output = self.main(inputs)
        b = inputs.size(0)
        return output.view(b, self.out_dim)


# DCGAN generator
class DCGAN_G(torch.nn.Module):
    def __init__(self, input_size, noise_dim, num_chan, 
                 hid_dim, ngpu=1, bn = False):
        super(DCGAN_G, self).__init__()
        self.__dict__.update(locals())
        self.register_buffer('device_id', torch.zeros(1))
        main = torch.nn.Sequential()
        # We need to know how many layers we will use at the beginning
        mult = input_size // 8
        ### Start block
        # Z_size random numbers
        
        main.add_module('Start-ConvTranspose2d', torch.nn.ConvTranspose2d(noise_dim, hid_dim * mult, kernel_size=4, stride=1, padding=0, bias=False))
        main.add_module('Start-lkrelu', torch.nn.SELU(inplace=True))
        
        #main.add_module('Start-conv-1x1', torch.nn.Conv2d(hid_dim * mult//2, hid_dim * mult, kernel_size= 1, stride=1, padding=0, bias=True))
        #main.add_module('Start-mid-lkrelu', torch.nn.LeakyReLU(0.2, inplace=True))
        
        # Size = (G_h_size * mult) x 4 x 4
        ### Middle block (Done until we reach ? x image_size/2 x image_size/2)
        i = 1
        while mult > 1:
            
            
            main.add_module('Middle-ConvTranspose2d [%d]' % i, torch.nn.ConvTranspose2d(hid_dim * mult, hid_dim * (mult//2), kernel_size=4, 
                                                            stride=2, padding=1, bias=False))
            main.add_module('Middle-mid-SELU [%d]' % i,  torch.nn.SELU(inplace=True))
            #main.add_module('Middle-mid-Norm [%d]' % i,  torch.nn.BatchNorm2d(hid_dim * mult))
            if bn is True:
                main.add_module('Middle-mid-Norm[%d]'%i,  torch.nn.BatchNorm2d( hid_dim * (mult//2)) )
            else:
                main.add_module('Middle-mid-Norm[%d]'%i, LayerNormal( hid_dim * (mult//2)) )
                #pass
                
            main.add_module('Middle-mid-drop [%d]' % i, torch.nn.Dropout2d(0.2)) 
            
            #main.add_module('Middle-mid-conv [%d]' % i, torch.nn.Conv2d(hid_dim * (mult//2), hid_dim * (mult//2), kernel_size=3, stride=1, padding=1, bias=True))
            #main.add_module('Middle-BatchNorm2d [%d]' % i, torch.nn.BatchNorm2d(hid_dim * (mult//2)))
            #main.add_module('Middle-SELU [%d]' % i, torch.nn.SELU(inplace=True))
            # Size = (G_h_size * (mult/(2*i))) x 8 x 8
            mult = mult // 2
            i += 1
        
        ### End block
        # Size = G_h_size x image_size/2 x image_size/2
        main.add_module('End-ConvTranspose2d', torch.nn.ConvTranspose2d(hid_dim, num_chan, kernel_size=4, stride=2, padding=1, bias=False))
        #main.add_module('End-Norm [%d]' % i, LayerNormal(hid_dim ))
        #main.add_module('End-SELU [%d]' % i, torch.nn.LeakyReLU(0.2, inplace=True))  
        
        #main.add_module('End-end-conv [%d]' % i, torch.nn.Conv2d(hid_dim, num_chan, kernel_size=3, stride=1, padding=1, bias=True))
        main.add_module('End-Tanh', torch.nn.Tanh())
        # Size = n_colors x image_size x image_size
        self.main = main

        self.apply(weights_init)

    def forward(self, inputs):
        inputs = inputs.view(inputs.size()[0], self.noise_dim, 1, 1)
        if isinstance(inputs.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = torch.nn.parallel.data_parallel(self.main, inputs, range(self.ngpu))
        else:
            output = self.main(inputs)
        return output

# DCGAN discriminator (using somewhat the reverse of the generator)
# Removed Batch Norm we can't backward on the gradients with BatchNorm2d
class DCGAN_D(torch.nn.Module):
    def __init__(self, input_size, num_chan, hid_dim, out_dim = 1, ngpu=1,  bn=False):
        super(DCGAN_D, self).__init__()
        self.register_buffer('device_id', torch.zeros(1))
        self.__dict__.update(locals())

        main = torch.nn.Sequential()
        ### Start block
        # Size = n_colors x image_size x image_size
        main.add_module('Start-Conv2d', torch.nn.Conv2d(num_chan,  hid_dim, kernel_size=4, stride=2, padding=1, bias=True))
        main.add_module('Start-SELU [%d]', torch.nn.SELU(inplace=True))
        
        #main.add_module('Start-mid-conv', torch.nn.Conv2d(hid_dim, hid_dim, kernel_size=3, stride=1, padding=1, bias=True))
        #main.add_module('Start-mid-SELU', torch.nn.LeakyReLU(0.2, inplace=True))
        
        image_size_new = input_size // 2
        # Size = D_h_size x image_size/2 x image_size/2
        
        ### Middle block (Done until we reach ? x 4 x 4)
        mult = 1
        i = 0
        while image_size_new > 4:
            
            main.add_module('Middle-Conv2d [%d]' % i, torch.nn.Conv2d(hid_dim * mult, hid_dim * (2*mult), kernel_size=4, stride=2, padding=1, bias=True))
            main.add_module('Middle-mid-SELU [%d]' % i, torch.nn.SELU(inplace=True))
            if bn is True:
                main.add_module('Middle-mid-Norm[%d]'%i,  torch.nn.BatchNorm2d(hid_dim * (2*mult)) )
            else:
                main.add_module('Middle-mid-Norm[%d]'%i, LayerNormal(hid_dim * (2*mult)) )
                #pass
            
            
            main.add_module('Middle-mid-drop [%d]' % i, torch.nn.Dropout2d(0.2)) 
            # Size = (D_h_size*(2*i)) x image_size/(2*i) x image_size/(2*i)
            image_size_new = image_size_new // 2
            mult = mult*2
            i += 1
            
        ### End block
        # Size = (D_h_size * mult) x 4 x 4
        main.add_module('End-Conv2d', torch.nn.Conv2d(hid_dim * mult, out_dim, kernel_size = 4, stride=1, padding=0, bias=True))     
        #main.add_module('Middle-LayerNorm2d[%d]'%i, LayerNormal( hid_dim * mult ) )
        #main.add_module('End-LeakyReLU [%d]' % i, torch.nn.LeakyReLU(0.2, inplace=True))
        #main.add_module('Middle-BatchNorm2d [%d]' % i, torch.nn.BatchNorm2d( hid_dim * mult ))
        #main.add_module('Middle-LeakyReLU [%d]' % i, torch.nn.LeakyReLU(0.2, inplace=True))
        #main.add_module('End-end-Conv2d', torch.nn.Conv2d(hid_dim * mult, out_dim, kernel_size = 1, stride=1, padding=0, bias=True))    
        
        self.main = main
        self.apply(weights_init)
        
    def forward(self, inputs):
        if isinstance(inputs.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = torch.nn.parallel.data_parallel(self.main, inputs, range(self.ngpu))
        else:
            output = self.main(inputs)
             
        b = output.size()[0]
        output = output.view(b, self.out_dim, -1)
        output = torch.mean(output, -1) #(b, out_dim)
        #output = torch.mean(output, 0) #(out_dim)
        # From batch_size x 1 x 1 (DCGAN used the sigmoid instead before)
        # Convert from batch_size x 1 x 1 to batch_size
        return output.view(b, self.out_dim)


class mmdNetG(nn.Module):
    def __init__(self, decoder):
        super(mmdNetG, self).__init__()
        self.register_buffer('device_id', torch.zeros(1))
        self.decoder = decoder

    def forward(self, inputs):
        output = self.decoder(inputs)
        return output


# NetD is an encoder + decoder
# input: batch_size * nc * image_size * image_size
# f_enc_X: batch_size * k * 1 * 1
# f_dec_X: batch_size * nc * image_size * image_size
class mmdNetD(nn.Module):
    def __init__(self, encoder, decoder):
        super(mmdNetD, self).__init__()
        self.register_buffer('device_id', torch.zeros(1))
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, inputs):
        f_enc_X = self.encoder(inputs)
        f_dec_X = self.decoder(f_enc_X)

        #f_enc_X = f_enc_X.view(input.size(0), -1)
        #f_dec_X = f_dec_X.view(input.size(0), -1)
        return f_enc_X, f_dec_X


## Weights init function, DCGAN use 0.02 std
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        # Estimated variance, must be around 1
        m.weight.data.normal_(1.0, 0.02)
        # Estimated mean, must be around 0
        m.bias.data.fill_(0)
