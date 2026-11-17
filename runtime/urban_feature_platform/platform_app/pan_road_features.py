"""Location-free road geometry features for a learned false-positive verifier."""

import cv2
import numpy as np


def road_structure_features(image, road_probability):
    image=np.asarray(image,np.float32)
    features=[image,road_probability,cv2.blur(road_probability,(7,7))]
    for size in (3,9,21):
        mean=cv2.blur(image,(size,size))
        std=np.sqrt(np.maximum(cv2.blur(image*image,(size,size))-mean*mean,0))
        features.extend([mean,std,image-mean])
    # Scale-normalized Hessian eigenvalues distinguish thin ridges from blobs.
    for sigma in (1.,2.,4.):
        smooth=cv2.GaussianBlur(image,(0,0),sigma)
        xx=cv2.Sobel(smooth,cv2.CV_32F,2,0,ksize=3)*sigma*sigma
        yy=cv2.Sobel(smooth,cv2.CV_32F,0,2,ksize=3)*sigma*sigma
        xy=cv2.Sobel(smooth,cv2.CV_32F,1,1,ksize=3)*sigma*sigma
        delta=np.sqrt((xx-yy)**2+4*xy*xy)
        a,b=(xx+yy-delta)/2,(xx+yy+delta)/2
        large=np.where(np.abs(a)>np.abs(b),a,b)
        small=np.where(np.abs(a)>np.abs(b),b,a)
        ratio=np.abs(small)/(np.abs(large)+1e-5)
        features.extend([large,ratio])
    gx=cv2.Sobel(image,cv2.CV_32F,1,0,ksize=3)
    gy=cv2.Sobel(image,cv2.CV_32F,0,1,ksize=3)
    xx,yy,xy=cv2.blur(gx*gx,(13,13)),cv2.blur(gy*gy,(13,13)),cv2.blur(gx*gy,(13,13))
    coherence=np.sqrt((xx-yy)**2+4*xy*xy)/(xx+yy+1e-5)
    features.extend([coherence,np.sqrt(gx*gx+gy*gy)])
    return np.stack(features,axis=-1).astype(np.float32)
