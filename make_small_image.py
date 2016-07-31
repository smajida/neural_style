import os
import mxnet as mx
import numpy as np
import symbol
from skimage import io, transform


def crop_img(im, size):
    im = io.imread(im)
    if im.shape[0] > im.shape[1]:
        c = (im.shape[0]-im.shape[1]) / 2
        im = im[c:c+im.shape[1],:,:]
    else:
        c = (im.shape[1]-im.shape[0]) / 2
        im = im[:,c:c+im.shape[0],:]
    im = transform.resize(im, (size,size))
    im *= 255
    return im

def preprocess_img(im, size):
    im = crop_img(im, size)
    im = np.swapaxes(im, 0, 2)
    im = np.swapaxes(im, 1, 2)
    im[0,:] -= 123.68
    im[1,:] -= 116.779
    im[2,:] -= 103.939
    im = np.expand_dims(im, 0)
    return im

def postprocess_img(im, color_ref=None):
    im = im[0]
    im[0,:] += 123.68
    im[1,:] += 116.779
    im[2,:] += 103.939
    im = np.swapaxes(im, 0, 2)
    im = np.swapaxes(im, 0, 1)
    im[im<0] = 0
    im[im>255] = 255
    return im.astype(np.uint8)


def make_image(img, style, save_name, noise, size=512):
    generator = symbol.generator_symbol()
    args = mx.nd.load('models/args%s_style.nd'%style)
    for i in range(6):
        args['znoise_%d'%i] = mx.nd.zeros([1,1,size/32*2**i,size/32*2**i], mx.gpu())
        args['zim_%d'%i] = mx.nd.zeros([1,3,size/32*2**i, size/32*2**i], mx.gpu())
    gene_executor = generator.bind(ctx=mx.gpu(), args=args, aux_states=mx.nd.load('models/auxs%s_style.nd'%style))
    for i in range(6):
        gene_executor.arg_dict['zim_%d'%i][:] = preprocess_img(img, size/32*2**i)
        gene_executor.arg_dict['znoise_%d'%i][:] = np.random.uniform(-noise,noise,[1,1,size/32*2**i,size/32*2**i])
    gene_executor.forward(is_train=True)
    out = gene_executor.outputs[0].asnumpy()
    im = postprocess_img(out)
    io.imsave(save_name, im)

def test(noise=250, size=512):
    for img in os.listdir('test_pics'):
        for style in ['8','9','10']:
            print style, img
            make_image('test_pics/%s'%img, style, 'out/small_%s_%s'%(style, img), noise, size)



