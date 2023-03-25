# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 20:44:28 2022

@author: DongXiao
"""

import matplotlib.pylab as plt
import functools
import time

def plot_image(im, title = 'img', colormap = 'gray',min_value = 0, max_value = 10):
    plt.figure(figsize=(5,5))
    plt.imshow(im,cmap= colormap,vmin = min_value, vmax = max_value), 
    plt.axis('off')
    plt.colorbar(fraction=0.045)
    plt.title(title, size=20)
    
def plot_single_image(im, title = 'img', colormap = 'gray',min_value = 0, max_value = 10):
    plt.imshow(im,cmap= colormap,vmin = min_value, vmax = max_value), 
    plt.axis('off')
    plt.colorbar(fraction=0.045)
    plt.title(title, size=20)

  
def plot_images(img1, img2, img3, title = ['img1','img2','img3'], colormap = 'nipy_spectral', min_v=0, max_v=4):
    plt.figure(figsize=(15,5))
    plt.gray()
    plt.subplot(131) 
    plot_single_image(img1, title[0], colormap= colormap,min_value = min_v, max_value = max_v)
    plt.subplot(132) 
    plot_single_image(img2, title[1], colormap= colormap,min_value = min_v, max_value = max_v)
    plt.subplot(133) 
    plot_single_image(img3, title[2], colormap= colormap,min_value = min_v, max_value = max_v)
    plt.tight_layout()
    plt.show()
    
    


def timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()    
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      
        run_time = end_time - start_time
        print(f"Complete {func.__name__!r} in {run_time:.4f} secs")
        return value
    return wrapper_timer