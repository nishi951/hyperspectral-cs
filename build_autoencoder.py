import torch
import torch.nn as nn
import torch.cuda as cuda
from collections import OrderedDict

def build_autoencoder(height, width, inchannels, outchannels, filtersize, nlayers):
    samepad = int((filtersize-1)/2)

    ###########
    # Encoder #
    ###########
    encoderLayers = [nn.Conv2d(inchannels, outchannels, filtersize, padding=samepad)]
    for i in range(nlayers-1):
        encoderLayers += [nn.ReLU(), nn.Conv2d(outchannels, outchannels, filtersize, padding=samepad)]
    Encoder = nn.Sequential(*encoderLayers)

    ###########
    # Decoder #
    ########### 
    decoderLayers = [] 
    for i in range(nlayers-1):
        decoderLayers += [nn.ReLU(), nn.Conv2d(outchannels, outchannels, filtersize, padding=samepad)]
    decoderLayers += [nn.ReLU(), nn.Conv2d(outchannels, inchannels, filtersize, padding=samepad), nn.ReLU()]
    Decoder = nn.Sequential(*decoderLayers)

    ###############
    # Autoencoder #
    ###############
    Autoencoder = nn.Sequential(OrderedDict([
        ("Encoder", Encoder),
        ("Decoder", Decoder)])
    )
    return Autoencoder