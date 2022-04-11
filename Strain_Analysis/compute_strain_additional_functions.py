import numpy as np
import os 
import matplotlib.pyplot as plt
from skimage import io
import cv2
import glob 
from scipy.signal import find_peaks
from scipy.spatial.distance import pdist, squareform
import av
import imageio
import moviepy.editor as mp
##########################################################################################
# create a mask 
##########################################################################################
def define_mask(cx,cy,center_w,center_h,image):
	mask_1_L = int(cx-center_w/2.0)
	mask_1_U = int(cx+center_w/2.0)
	mask_0_L = int(cy-center_h/2.0)
	mask_0_U = int(cy+center_h/2.0)

	mask = np.zeros(image.shape)
	mask[mask_0_L:mask_0_U,mask_1_L:mask_1_U] = 1
	mask = mask.astype('uint8')
	
	return mask 
	
##########################################################################################
# process the input tifs so that the tracking algorithm works 
##########################################################################################
def process_tif_color_scale(input_tag,tag,mask):
	npy_folder = tag + '_npy_arrays'
	if not os.path.exists(npy_folder):
		os.mkdir(npy_folder)

	im = io.imread( input_tag + '.tif')
	
	vec = [] 
	for frame_kk in range(0,im.shape[0]):
		frame_npy = im[frame_kk,:,:]
		for kk in range(0,mask.shape[0]):
			for jj in range(0,mask.shape[1]):
				if mask[kk,jj] > 0:
					vec.append(frame_npy[kk,jj])
	
	vec_sorted = np.sort(vec)		
	vec_len = len(vec)		
	max_val = vec_sorted[int(vec_len-1)]
	min_val = vec_sorted[int(0)]
	
	print(max_val, min_val)
	
	for frame_kk in range(0,im.shape[0]):
		frame_npy = im[frame_kk,:,:]
		# ------------ color correction --------------------------------------------------
		fim_bounds = np.zeros(frame_npy.shape)
		for kk in range(0,frame_npy.shape[0]):
			for jj in range(0,frame_npy.shape[1]):
				if frame_npy[kk,jj] > max_val:
					fim_bounds[kk,jj] = max_val
				elif frame_npy[kk,jj] < min_val:
					fim_bounds[kk,jj] = min_val
				else:
					fim_bounds[kk,jj] = frame_npy[kk,jj]
		
		frame_npy = fim_bounds 
		# --------------------------------------------------------------------------------
		plt.imsave(npy_folder + '/' + tag + '_%04d.png'%(frame_kk),frame_npy,cmap='gray')
		frame_npy = plt.imread(npy_folder + '/' + tag + '_%04d.png'%(frame_kk))		
		os.system('rm ' + npy_folder + '/' + tag + '_%04d.png'%(frame_kk))
		frame_npy = frame_npy[:,:,0]*255
		np.save(npy_folder + '/' + tag + '_%04d.npy'%(frame_kk),frame_npy)
		if frame_kk == 1:
			plt.imsave(tag + '_single_image.png',frame_npy,cmap='gray')
	
	return	


##########################################################################################
# process the input tifs so that the tracking algorithm works 
##########################################################################################
def make_movie_from_npy(input_tag,tag,include_eps=False): 
	npy_folder = tag + '_visualize'
	if not os.path.exists(npy_folder):
		os.mkdir(npy_folder)
	
	img_list = [] 
	num_frames = len(glob.glob(tag + '_npy_arrays/*'))
	for kk in range(0,num_frames):
		raw_img = np.load(tag + '_npy_arrays' + '/' + tag + '_%04d.npy'%(kk))
		plt.figure()
		plt.imshow(raw_img, cmap=plt.cm.gray)
		ax = plt.gca()
		ax.set_xticks([]); ax.set_yticks([])
		plt.savefig(npy_folder + '/' + 'frame_%04d.png'%(kk),bbox_inches = 'tight', pad_inches = 0)
		if include_eps:
			plt.savefig(npy_folder + '/' + 'frame_%i.eps'%(kk),bbox_inches = 'tight', pad_inches = 0)
		plt.close()
		img_list.append(imageio.imread(npy_folder + '/' + 'frame_%04d.png'%(kk)))
	
	imageio.mimsave(npy_folder + '/contract_anim.gif', img_list)	
	clip = mp.VideoFileClip(npy_folder + '/contract_anim.gif')
	clip.write_videofile( npy_folder + '/contract_anim.mp4')
	
	return
	
##########################################################################################
# compute the break points 
##########################################################################################
def compute_frame_pairs(tag,mask):
	first_frame_idx = 0
	window = 10
	folder_data = tag + '_npy_arrays'
	li = glob.glob(folder_data + '/*')
	num_frames = len(li) - first_frame_idx
	feature_params =  dict( maxCorners = 10000, qualityLevel = 0.01, minDistance = 2, blockSize = 5)
	lk_params = dict( winSize  = (window, window),
					  maxLevel = 10,
					  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

	# ------------------------------------------------------------------------------------
	# load first frame to determine good quality features to track, set up npy arrays 
	# ------------------------------------------------------------------------------------
	img = np.load(tag + '_npy_arrays/' + tag + '_%04d.npy'%(first_frame_idx)).astype('uint8')
	p0 = cv2.goodFeaturesToTrack(img, mask = mask, **feature_params)
	num_mark = p0.shape[0]
	tracker_x = np.zeros((num_mark,num_frames))
	tracker_y = np.zeros((num_mark,num_frames))
	tracker_x[:,0] = p0[:,0,0]
	tracker_y[:,0] = p0[:,0,1]

	# ------------------------------------------------------------------------------------
	# run tracking 
	# ------------------------------------------------------------------------------------
	for kk in range(1,num_frames):
		img_old = img.copy()
		img = np.load(tag + '_npy_arrays/' + tag + '_%04d.npy'%(kk + first_frame_idx)).astype('uint8')
		p1, st, err = cv2.calcOpticalFlowPyrLK(img_old, img, p0, None, **lk_params)
		tracker_x[:,kk] = p1[:,0,0]
		tracker_y[:,kk] = p1[:,0,1]
		p0 = p1.copy()

	pos_x_all = tracker_x
	pos_y_all = tracker_y

	num_frames = pos_x_all.shape[1]
	num_pts = pos_x_all.shape[0]

	tracking_x_all = np.zeros(pos_x_all.shape)
	tracking_y_all = np.zeros(pos_y_all.shape)

	for kk in range(0,num_frames):
		tracking_x_all[:,kk] = pos_x_all[:,kk] - pos_x_all[:,0]
		tracking_y_all[:,kk] = pos_y_all[:,kk] - pos_y_all[:,0]

	tracking_abs_all = ((tracking_x_all)**2.0 + (tracking_y_all)**2.0)**0.5
	mean_abs_all = np.mean(tracking_abs_all, axis=0) # mean absolute distance 
	
	# ------------------------------------------------------------------------------------
	# compute break points
	# ------------------------------------------------------------------------------------
	# find the peaks of the mean curve 
	# compute average distance to nearest neighbor for each frame 

	peaks_abs, _ = find_peaks(mean_abs_all,distance=20,prominence=0.1)
	valleys_abs, _ = find_peaks(-1*mean_abs_all,distance=20)

	reference_points = []
	reference_points.append(0) # frame 0 is a reference point 

	for kk in range(0,peaks_abs.shape[0]-1):
		reference_points.append(int(peaks_abs[kk]*0.5 + peaks_abs[kk+1]*0.5))

	reference_points = np.asarray(reference_points)
	
	if False: # used for de-bugging 
		plt.figure()
		plt.plot(mean_abs_all)
	
	frame_pairs = [] 
	for kk in range(1,reference_points.shape[0]-1):
		frame_pairs.append([reference_points[kk],reference_points[kk+1]])
		val1 = reference_points[kk]
		val2 = reference_points[kk+1]
		if False:
			plt.plot([val1,val2],[mean_abs_all[val1],mean_abs_all[val2]],'ro')
	
	if False:
		plt.savefig(tag + '_abs_tracking.png')
	
	return frame_pairs


##########################################################################################
# track a limited block
##########################################################################################
def lk_block_track(tag,mask,frame_0,block_size,window=10):
	feature_params =  dict( maxCorners = 10000, qualityLevel = 0.01, minDistance = 2, blockSize = 5)
	lk_params = dict( winSize  = (window, window),
					  maxLevel = 10,
					  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
	
	img = np.load(tag + '_npy_arrays/' + tag + '_%04d.npy'%(frame_0)).astype('uint8')
	p0 = cv2.goodFeaturesToTrack(img, mask = mask, **feature_params)
	num_mark = p0.shape[0]
	tracker_x = np.zeros((num_mark,block_size))
	tracker_y = np.zeros((num_mark,block_size))
	tracker_x[:,0] = p0[:,0,0]
	tracker_y[:,0] = p0[:,0,1]
	
	for kk in range(1,block_size):
		img_old = img.copy()
		img = np.load(tag + '_npy_arrays/' + tag + '_%04d.npy'%(frame_0 + kk)).astype('uint8')
		p1, st, err = cv2.calcOpticalFlowPyrLK(img_old, img, p0, None, **lk_params)
		tracker_x[:,kk] = p1[:,0,0]
		tracker_y[:,kk] = p1[:,0,1]
		p0 = p1.copy()

	return tracker_x, tracker_y

##########################################################################################
# compute F
##########################################################################################
def compute_F(Lambda_0,Lambda_t):
	term_1 = np.dot( Lambda_t , np.transpose(Lambda_0) )
	term_2 = np.linalg.inv( np.dot( Lambda_0 , np.transpose(Lambda_0) ) )
	F = np.dot(term_1 , term_2)
	return F 

##########################################################################################
# compute strain (calls lk_block_track and compute_F)
##########################################################################################
def compute_strain(tracker_x,tracker_y):
	
	num_pts = tracker_x.shape[0]
	num_frames = tracker_x.shape[1]
	
	Lambda_0 = []; step = 0 
	for kk in range(0,num_pts):
		for jj in range(kk+1,num_pts):
			Lambda_0.append([ tracker_x[kk,step] - tracker_x[jj,step], tracker_y[kk,step] - tracker_y[jj,step] ])
	
	Lambda_0 = np.asarray(Lambda_0).T
	
	F_all = [] 
	for t in range(0,num_frames):
		Lambda_t = []; step = t
		for kk in range(0,num_pts):
			for jj in range(kk+1,num_pts):
				Lambda_t.append([ tracker_x[kk,step] - tracker_x[jj,step], tracker_y[kk,step] - tracker_y[jj,step] ])
		
		Lambda_t = np.asarray(Lambda_t).T
		F = compute_F(Lambda_0,Lambda_t)
		F_all.append(F)
	
	Exx_all = [] 
	Eyy_all = [] 
	Exy_all = [] 
	for kk in range(0,num_frames):
		F = F_all[kk]
		C = np.dot(F.T,F)
		E = 0.5*(C - np.eye(2))
		Exx_all.append(E[0,0])
		Eyy_all.append(E[1,1])
		Exy_all.append(E[0,1])

	return F_all, Exx_all, Eyy_all, Exy_all





