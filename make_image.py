import os
import time
import mxnet as mx
import numpy as np
import symbol
from skimage import io, transform


def crop_img(im, size, scale):
    im = io.imread(im)
    s0, s1, _ = im.shape
    c0 = (s0-size[0]) / 2
    c1 = (s1-size[1]) / 2
    im = im[c0:c0+s0,c1:c1+s1]
    im = transform.resize(im, [size[0]*scale, size[1]*scale])
    im = im*255
    return im

def preprocess_img(im, size, scale):
    im = crop_img(im, size, scale)
    im = im.astype(np.float32)
    im = np.swapaxes(im, 0, 2)
    im = np.swapaxes(im, 1, 2)
    im[0,:] -= 123.68
    im[1,:] -= 116.779
    im[2,:] -= 103.939
    im = np.expand_dims(im, 0)
    return im

def postprocess_img(im):
    im = im[0]
    im[0,:] += 123.68
    im[1,:] += 116.779
    im[2,:] += 103.939
    im = np.swapaxes(im, 0, 2)
    im = np.swapaxes(im, 0, 1)
    im[im<0] = 0
    im[im>255] = 255
    return im.astype(np.uint8)


def make_image(img, style, save_name, noise):
    generator = symbol.generator_symbol()
    args = mx.nd.load('models/args%s_style.nd'%style)
    s0, s1, _ = io.imread(img).shape
    s0 = s0/32*32
    s1 = s1/32*32
    for i in range(6):
        args['znoise_%d'%i] = mx.nd.zeros([1,1,s0/32*2**i,s1/32*2**i], mx.gpu())
        args['zim_%d'%i] = mx.nd.zeros([1,3,s0/32*2**i, s1/32*2**i], mx.gpu())
    gene_executor = generator.bind(ctx=mx.gpu(), args=args, aux_states=mx.nd.load('models/auxs%s_style.nd'%style))
    for i in range(6):
        gene_executor.arg_dict['zim_%d'%i][:] = preprocess_img('%s'%img, [s0, s1], 1./(2**(5-i)))
        gene_executor.arg_dict['znoise_%d'%i][:] = np.random.uniform(-noise,noise,[1,1,s0/32*2**i,s1/32*2**i])
    gene_executor.forward(is_train=True)
    out = gene_executor.outputs[0].asnumpy()
    im = postprocess_img(out)
    io.imsave(save_name, im)


def test(noise):
    for img in os.listdir('test_pics'):
        for style in ['8','9','10']:
            print style, img
            make_image('test_pics/%s'%img, style, 'out/%s_%s'%(style, img), noise)
