import numpy as np
from skimage import io

# convert image to 10 small images 
# img = io.imread('D1T3_Before.tif')
# 
# num_frames = img.shape[0]
# 
# num_split = 10 
# 
# frames_per_split = int(num_frames / num_split)
# 
# for kk in range(0,num_split):
# 	ix0 = kk*frames_per_split
# 	ix1 = (kk+1)*frames_per_split
# 	if ix1 > num_frames:
# 		ix1 = num_frames
# 	
# 	img_subset = img[ix0:ix1,:,:]
# 	io.imsave('D1T3_Before_frame%i_to_frame%i.tif'%(ix0,ix1-1),img_subset)

# re-combine the image into a single image 
num_split = 10
frames_per_split = 20 
num_frames = 200
dim1 = 512
dim2 = 512

img = np.zeros((num_frames,dim1,dim2))

for kk in range(0,num_split):
	ix0 = kk*frames_per_split
	ix1 = (kk+1)*frames_per_split -1 
	img_slice = io.imread('D1T3_Before_frame%i_to_frame%i.tif'%(ix0,ix1))
	img[ix0:(ix1+1),:,:] = img_slice

img = img.astype('uint16')

io.imsave('D1T3_Before.tif',img)

