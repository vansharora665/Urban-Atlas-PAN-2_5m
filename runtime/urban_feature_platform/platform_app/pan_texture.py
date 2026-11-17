"""Location-free, multi-scale PAN texture features with bounded spatial context."""

import cv2
import numpy as np

FEATURES=["intensity","mean3","std3","mean7","std7","mean15","std15",
          "mean31","std31","mean63","std63","gradient","gradient_mean7",
          "gradient_mean31","relative_texture15","relative_texture63"]
HALO=32


def texture_features(image):
    image=np.ascontiguousarray(image,dtype=np.float32)
    features=[image]
    for size in (3,7,15,31,63):
        mean=cv2.blur(image,(size,size),borderType=cv2.BORDER_REFLECT_101)
        variance=np.maximum(cv2.blur(image*image,(size,size),borderType=cv2.BORDER_REFLECT_101)-mean*mean,0)
        features.extend((mean,np.sqrt(variance)))
    gx=cv2.Sobel(image,cv2.CV_32F,1,0,ksize=3)
    gy=cv2.Sobel(image,cv2.CV_32F,0,1,ksize=3)
    gradient=np.hypot(gx,gy)
    features.extend((gradient,cv2.blur(gradient,(7,7)),cv2.blur(gradient,(31,31)),
                     features[6]/np.maximum(features[5],.1),features[10]/np.maximum(features[9],.1)))
    return np.stack(features,axis=-1).astype(np.float32)
