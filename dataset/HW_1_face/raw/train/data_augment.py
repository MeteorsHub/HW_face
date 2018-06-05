from PIL import Image  
from skimage import data, exposure, img_as_float
import os
import numpy as np

dirname = os.path.dirname(os.path.abspath(__file__))

for i in range(2000):
	im = Image.open(dirname + '/%d/%d_1.jpg' % (i,i))  
	gam1 = exposure.adjust_gamma(im, 2)
	gam2 = exposure.adjust_gamma(im, 0.5)
	im_Contrast = skimage.exposure.rescale_intensity(im)
	gam1.save(dirname + '/%d/%d_2.jpg' % (i,i))
	gam2.save(dirname + '/%d/%d_3.jpg' % (i,i))
	im_Contrast.save(dirname + '/%d/%d_4.jpg' % (i,i))
	im_noise = im
	rows,cols,dims=im_noise.shape
	for j in range(500):    
		x=np.random.randint(0,rows)
		y=np.random.randint(0,cols)
		im_noise[x,y,:]=255
	im_noise.save(dirname + '/%d/%d_5.jpg' % (i,i))


	im_transpose = im.transpose(Image.FLIP_LEFT_RIGHT)
	im_transpose.save(dirname + '/%d/%d_6.jpg' % (i,i))
	gam1 = exposure.adjust_gamma(im_transpose, 2)
	gam2 = exposure.adjust_gamma(im_transpose, 0.5)
	im_Contrast = skimage.exposure.rescale_intensity(im_transpose)
	gam1.save(dirname + '/%d/%d_7.jpg' % (i,i))
	gam2.save(dirname + '/%d/%d_8.jpg' % (i,i))
	im_Contrast.save(dirname + '/%d/%d_9.jpg' % (i,i))
	im_noise = im_transpose
	rows,cols,dims=im_noise.shape
	for j in range(500):    
		x=np.random.randint(0,rows)
		y=np.random.randint(0,cols)
		im_noise[x,y,:]=255
	im_noise.save(dirname + '/%d/%d_10.jpg' % (i,i))
