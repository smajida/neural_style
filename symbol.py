import mxnet as mx
import numpy as np

def block(data, num_filter, name):
    data = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(3,3), pad=(1,1), name='%s_conv1'%name)
    data = mx.sym.CuDNNBatchNorm(data=data, momentum=0, name='%s_batchnorm1'%name)
    data = mx.sym.LeakyReLU(data=data, slope=0.1)
    data = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(3,3), pad=(1,1), name='%s_conv2'%name)
    data = mx.sym.CuDNNBatchNorm(data=data, momentum=0, name='%s_batchnorm2'%name)
    data = mx.sym.LeakyReLU(data=data, slope=0.1)
    data = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=(1,1), pad=(0,0), name='%s_conv3'%name)
    data = mx.sym.CuDNNBatchNorm(data=data, momentum=0, name='%s_batchnorm3'%name)
    data = mx.sym.LeakyReLU(data=data, slope=0.1)
    return data


def join(data, data_low, num_filter, name):
    data_low = mx.sym.UpSampling(data_low, scale=2, num_filter=num_filter, sample_type='nearest', num_args=1)
#     data_low = mx.sym.Deconvolution(data_low, kernel=(2,2), stride=(2,2), num_filter=num_filter, name='%s_upsample_low'%name)
    data_low = mx.sym.CuDNNBatchNorm(data=data_low, momentum=0, name='%s_batchnorm_low'%name)
    data = mx.sym.CuDNNBatchNorm(data=data, momentum=0, name='%s_batchnorm'%name)
    out = mx.sym.Concat(data, data_low)
    return out


def generator_symbol():
    Z = []
    for i in range(6):
        noise = mx.sym.Variable('znoise_%d'%i)
        im = mx.sym.Variable('zim_%d'%i)
        Z.append(block(mx.sym.Concat(noise, im), 8, name='block%d'%i))
    for i in range(1,6):
        Z[i] = block(join(Z[i], Z[i-1], i*8, name='join%d'%i), 8*(i+1), name='blockjoin%d'%i)
    out = mx.sym.Convolution(data=Z[-1], num_filter=3, kernel=(1,1), pad=(0,0), name='blockout')
    return out
