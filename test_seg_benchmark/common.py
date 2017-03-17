'''
live repetition counting system
Ofir Levy, Lior Wolf
Tel Aviv University
'''
import os
import numpy as np 
import theano
import theano.tensor as T
from scipy.ndimage import filters
import scipy
import cv2

from bluevelvet.image.resize import *


def shared_dataset(data_xy, borrow=True):
    data_x, data_y = data_xy
    shared_x = theano.shared(np.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(np.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    return shared_x, T.cast(shared_y, 'int32'), shared_y

def load_single_set(cFrames, set_num):

    xx = cFrames[0,20*set_num+0]
    datset = xx.reshape(1,xx.shape[0]*xx.shape[1])

    for i in range(1,20):
        xx = cFrames[0,20*set_num+i]
        xx = xx.reshape(1,xx.shape[0]*xx.shape[1])
        datset = np.append(datset, xx, axis=1)
    return datset


def load_initial_test_data():
    # just init gpu data with zeros
    data_x = np.zeros((5,50000), dtype=np.float)
    labels = np.zeros((1,5), dtype=np.uint8)
    data_x = data_x.astype(theano.config.floatX)
    data_y = labels
    data_y = data_y.reshape((data_y.shape[1]))
    data_y = data_y.astype(theano.config.floatX)
    train_set = data_x, data_y
    train_set_x, train_set_y, shared_train_set_y  = shared_dataset(train_set)
    rval = (train_set_x, train_set_y, shared_train_set_y, 1)
    return rval


def get_boundingbox(frame_set):
    '''
    :param frame_set: NumPy array (N,H,W) of type np.uint8 (0-255)
    :return: frame_rois: NumPy array with crops (N,50,50) of type np.float32 (0-1)
    '''

    print(frame_set.shape)
    print(frame_set.dtype)

    fstd = np.std(frame_set,axis=0)
    framesstd = np.mean(fstd)
    #th = framesstd  / 3
    th = framesstd
    #ones = np.ones(8)
    ones = np.ones(10)
    big_var = (fstd>th)

    if (framesstd==0): # no bb, take full frame
        frameROIRes = np.zeros([20,50,50])
        for i in range(20):
            frameROIRes[i,:,:] = scipy.misc.imresize(frame_set[i,:,:], size=(50,50),interp='bilinear')

        frameROIRes = np.reshape(frameROIRes, (1,frameROIRes.shape[0]*frameROIRes.shape[1]*frameROIRes.shape[2]))
        frameROIRes = frameROIRes.astype(np.float32)

        return (frameROIRes, framesstd)

    big_var = big_var.astype(np.float32)
    big_var = filters.convolve1d(big_var, ones, axis=0)
    big_var = filters.convolve1d(big_var, ones, axis=1)

    th2 = 80
    i,j = np.nonzero(big_var>th2)

    if (i.size > 0):

        si = np.sort(i)
        sj = np.sort(j)

        ll = si.shape[0]
        th1 = round(ll*0.03)
        th2 = np.floor(ll*0.98)
        y1 = si[th1]
        y2 = si[th2]
        x1 = sj[th1]
        x2 = sj[th2]

        # cut image ROI
        if (((x2-x1)>0) and ((y2-y1)>0)):
            framesRoi = frame_set[:,y1:y2,x1:x2]
        else:
            framesRoi = frame_set[:,:,:]
    else:
        framesRoi = frame_set[:,:,:]

    # debug - show ROI
    #cv2.namedWindow('ROI', cv2.WINDOW_NORMAL)
    #bla= scipy.misc.imresize(framesRoi[19,:,:], size=(200,200),interp='bilinear')
    #cv2.imshow('ROI', bla)

    # resize to 50x50
    frameROIRes = np.zeros([20,50,50])
    for i in range(20):
        frameROIRes[i,:,:] = scipy.misc.imresize(framesRoi[i,:,:], size=(50,50),interp='bilinear')

    #frameROIRes = frameROIRes / 255  # TODO - does this really nessacarry?
    print(frameROIRes.dtype)
    print(frameROIRes.shape)

    return (frameROIRes)


def video_cube_center_crop(frame_set, output_size=50, show_frames=False):
    frames_crop = np.zeros((len(frame_set), output_size, output_size), np.float32)
    for i in range(len(frame_set)):
        frame_gray_crop_float = resize_crop_center(frame_set[i], output_size, output_size)/255.0
        frames_crop[i] = frame_gray_crop_float

        if show_frames:
            cv2.imshow("frame", frames_crop[i])
            cv2.waitKey(10)

    return frames_crop


def load_next_test_data(fileName, stride, search_bounding_box=True):

    # TODO - load the video once and split it into strides.
    #cap = cv2.VideoCapture('Push_ups_active.avi')
    cap = cv2.VideoCapture(fileName)
    frm_cnt = -1

    #stride = 8
    framesList = []
    framesData = []

    while True:
        ret, frame = cap.read()
        if (ret == 0):
            break
        frm_cnt = frm_cnt + 1
        # take every nth frame
        if (frm_cnt%stride != 0):
            continue
        # convert to gray
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # sub-sample for performance

        # TODO: add function parameter to control the subsampling...
        #gray_frame = gray_frame[::3,::3]

        # Add the grayscale frame to FIFO queue
        framesList.append(gray_frame)

        if len(framesList)>20:
            
            # Remove the first frame
            framesList.pop(0)
            
            # List of 20 frames for processing
            framesArr = np.array(framesList)
            frames = None
            
            if search_bounding_box:
                frames = get_boundingbox(framesArr)
                frames = frames.astype(np.float32)
                frames /= 255
            else:
                frames = video_cube_center_crop(framesArr, output_size=50)

            # Rescale the array to one big dimension
            frames = np.reshape(frames, (1,frames.shape[0]*frames.shape[1]*frames.shape[2]))

            # Add the video block of size [20,50,50] (=> flattened)
            framesData.append(frames)

    cap.release()
    if (framesData == []):
        # stride too big for this video
        return(-1,0)
    framesData = np.array(framesData)
    framesData = np.squeeze(framesData,axis=1)
    rval = (framesData, 1)
    return rval