import torch
from torch.autograd import Variable
import torch.nn.functional as F
import PIL

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from projection_utils import make_grid_np
from style_utils import extract_features, content_loss, style_loss, tv_loss, gram_matrix
from image_utils import *


def style_transfer(cnn, content_image, style_image, image_size, style_size, content_layer,
                   content_weight, style_layers, style_weights, tv_weight, init_random=False, num_iters=300):
    """
    Run style transfer!
    
    Inputs:
    - cnn: The pre-trained network
    - content_image: filename of content image
    - style_image: filename of style image
    - image_size: size of smallest image dimension (used for content loss and generated image)
    - style_size: size of smallest style image dimension
    - content_layer: layer to use for content loss
    - content_weight: weighting on content loss
    - style_layers: list of layers to use for style loss
    - style_weights: list of weights to use for each layer in style_layers
    - tv_weight: weight of total variation regularization term
    - init_random: initialize the starting image to uniform random noise
    """
    # Infer cuda or not from cnn's dtype
    dtype = next(cnn.parameters()).type()

    # Extract features for the content image
    content_img = preprocess(PIL.Image.open(content_image), size=image_size)
    content_img_var = Variable(content_img.type(dtype))

    # Extract features for the style image
    style_img = preprocess(PIL.Image.open(style_image), size=style_size)
    style_img_var = Variable(style_img.type(dtype))
    feats = extract_features(style_img_var, cnn)
    style_targets = []
    for idx in style_layers:
        style_targets.append(gram_matrix(feats[idx].clone()))

    # Initialize output image to content image or noise
    if init_random:
        img = torch.Tensor(content_img.size()).uniform_(0, 1)
    else:
        img = content_img.clone().type(dtype)

    # We do want the gradient computed on our image!
    img_var = Variable(img, requires_grad=True)

    # Set up optimization hyperparameters
    initial_lr = 3.0
    decayed_lr = 0.1
    decay_lr_at = 180

    # Note that we are optimizing the pixel values of the image by passing
    # in the img_var Torch variable, whose requires_grad flag is set to True
    optimizer = torch.optim.Adam([img_var], lr=initial_lr)
    
    f, axarr = plt.subplots(1,2)
    axarr[0].axis('off')
    axarr[1].axis('off')
    axarr[0].set_title('Content Source Img.')
    axarr[1].set_title('Style Source Img.')
    axarr[0].imshow(deprocess(content_img.cpu().clone()))
    axarr[1].imshow(deprocess(style_img.cpu().clone()))
    plt.show()
    plt.figure()
    
    num_proj = 20
    for t in range(num_iters):
        if t < (num_iters - 10):
            img.clamp_(-1.5, 1.5)
        optimizer.zero_grad()
        
        loss_lst = []
        
        for i in range(num_proj):
            theta, phi = np.random.uniform(0, 2 * np.pi), np.random.uniform(0, 2 * np.pi)
            grid_size = (image_size[0], image_size[1], 3)
            coords = make_grid_np(theta, phi, 500, grid_size )
            coords_var = Variable(torch.from_numpy(coords.astype(np.float32))[None].type(dtype), requires_grad=False)

            proj_content_img_var = F.grid_sample(content_img_var, coords_var)
            feats = extract_features(proj_content_img_var, cnn)
            content_target = feats[content_layer].clone()
            
            proj_img_var = F.grid_sample(img_var, coords_var)
            feats = extract_features(proj_img_var, cnn)

            # Compute loss
            c_loss = content_loss(content_weight, feats[content_layer], content_target)
            s_loss = style_loss(feats, style_layers, style_targets, style_weights)
            t_loss = tv_loss(proj_img_var, tv_weight)
            loss_lst.append(c_loss + s_loss + t_loss)


        loss = sum(loss_lst)

        loss.backward()

        # Perform gradient descents on our image values
        if t == decay_lr_at:
            optimizer = torch.optim.Adam([img_var], lr=decayed_lr)
        optimizer.step()

        if t % 100 == 0:
            print('Iteration {}'.format(t))
            plt.axis('off')
            plt.imshow(deprocess(img.cpu().clone()))
            plt.show()
    print('Iteration {}'.format(t))
    plt.axis('off')
    plt.imshow(deprocess(img.cpu().clone()))
    plt.show()
    return img_var