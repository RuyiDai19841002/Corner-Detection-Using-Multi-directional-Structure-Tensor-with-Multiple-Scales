import cv2
import proposed
from pylab import *
im = cv2.imread(r".\17.bmp")
r=proposed.improved1(im, 10 ** 8.4)#298
figure()
gray()
imshow(im)
plot([p[1] for p in r],[p[0]for p in r],'w.')
axis('off')
show()