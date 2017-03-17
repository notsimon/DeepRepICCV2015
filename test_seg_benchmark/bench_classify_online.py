'''
live repetition counting system
Ofir Levy, Lior Wolf
Tel Aviv University
'''
import os
import re
import numpy as np
import cv2
import glob
import pickle
from common import *

import bluevelvet.utils
print(bluevelvet.__file__)
#import bluevelvet.utils.evaluation as evaluation


def get_inter_num(data, valid):
    (ns_test_set_x_st2,ns_test_set_x_st5,ns_test_set_x_st8) = data
    (valid_st2,valid_st5,valid_st8) = valid
    if (valid_st8 == 1):
        return ((ns_test_set_x_st8.shape[0]-6)/5)
    if (valid_st5 == 1):
        return ((ns_test_set_x_st8.shape[0]-21)/8)
    else:
        return ((ns_test_set_x_st8.shape[0]-81)/20)


def load_movie_data(fileName):

    (ns_test_set_x_st2, valid_st2) = load_next_test_data(fileName, 2)
    (ns_test_set_x_st5, valid_st5) = load_next_test_data(fileName, 5)
    (ns_test_set_x_st8, valid_st8) = load_next_test_data(fileName, 8)

    return ((ns_test_set_x_st2,ns_test_set_x_st5,ns_test_set_x_st8), (valid_st2,valid_st5,valid_st8))


def count_in_interval(classify, test_set_x, ns_test_set_x, frame_residue, start, end):

    assert start <= end
    if (start == end):
        return (0, 0, 0)

    test_set_x.set_value(ns_test_set_x, borrow=True)

    rep_counts = 0
    entropy = 0
    for i in range(start,end):
        output_label , pYgivenX = classify(i)
        pYgivenX[pYgivenX==0] = np.float32(1e-30) # hack to output valid entropy
        entropy = entropy - (pYgivenX*np.log(pYgivenX)).sum()
        output_label = output_label + 3 # moving from label to cycle length
        if (i == 0):
            rep_counts = 20 / output_label
            frame_residue = 20 % output_label
        else:
            frame_residue += 1
            if (frame_residue >= output_label):
                rep_counts += 1;
                frame_residue = 0;

    ave_entropy = entropy/(end-start)
    return (rep_counts, frame_residue, ave_entropy)



def initial_count(classify, test_set_x, data, valid):

    (valid_st2,valid_st5,valid_st8) = valid
    (ns_test_set_x_st2,ns_test_set_x_st5,ns_test_set_x_st8) = data

    # classify st_2 it is always valid
    (st2_count, st2_res, st2_entropy) = count_in_interval(classify, test_set_x, ns_test_set_x_st2, 0, 0, 81)  #100 - 19 etc.

    # check if st5 is valid. if not return st2 count
    if (valid_st5 == 1):
        (st5_count, st5_res, st5_entropy) = count_in_interval(classify, test_set_x, ns_test_set_x_st5, 0, 0, 21)
    else:
        st8_entropy = np.inf

    if (valid_st8 == 1):
        (st8_count, st8_res, st8_entropy) = count_in_interval(classify, test_set_x, ns_test_set_x_st8, 0, 0, 6)
    else:
        st8_entropy = np.inf


    winner = np.nanargmin(np.array([st2_entropy, st5_entropy, st8_entropy]))

    if (winner == 0):
        # winner is stride 2
        return (st2_count, (st2_res*2/2,st2_res*2/5, st2_res*2/8))
    if (winner == 1):
        # winner is stride 5
        return (st5_count, (st5_res*5/2,st5_res*5/5, st5_res*5/8))
    if (winner == 2):
        # winner is stride 8
        return (st8_count, (st8_res*8/2,st8_res*8/5, st8_res*8/8))



def get_next_count(classify, test_set_x, data, valid, global_count, curr_residue, start_frame):

    (valid_st2,valid_st5,valid_st8) = valid
    (ns_test_set_x_st2,ns_test_set_x_st5,ns_test_set_x_st8) = data
    (curr_residue_st2, curr_residue_st5, curr_residue_st8) = curr_residue

    # classify st_2 it is always valid
    (st2_count, st2_res, st2_entropy) = count_in_interval(classify, test_set_x, ns_test_set_x_st2, curr_residue_st2, (start_frame/2-19), (start_frame/2-19)+20)
    # check if st5 is valid. if not return st2 count
    if (valid_st5 == 1):
        (st5_count, st5_res, st5_entropy) = count_in_interval(classify, test_set_x, ns_test_set_x_st5, curr_residue_st5, (start_frame/5-19), (start_frame/5-19)+8)
    else:
        st5_entropy = np.inf

    if (valid_st8 == 1):
        (st8_count, st8_res, st8_entropy) = count_in_interval(classify, test_set_x, ns_test_set_x_st8, curr_residue_st8, (start_frame/8-19), (start_frame/8-19)+5)
    else:
        st8_entropy = np.inf

    winner = np.nanargmin(np.array([st2_entropy, st5_entropy, st8_entropy]))

    if (winner == 0):
        # winner is stride 2
        return (global_count + st2_count, (st2_res*2/2,st2_res*2/5, st2_res*2/8))
    if (winner == 1):
        # winner is stride 5
        return (global_count + st5_count, (st5_res*5/2,st5_res*5/5, st5_res*5/8))
    if (winner == 2):
        # winner is stride 8
        return (global_count + st8_count, (st8_res*8/2,st8_res*8/5, st8_res*8/8))

def get_remain_count(classify, test_set_x, data, valid, global_count, curr_residue, start_frame):

    (valid_st2,valid_st5,valid_st8) = valid
    (ns_test_set_x_st2,ns_test_set_x_st5,ns_test_set_x_st8) = data
    (curr_residue_st2, curr_residue_st5, curr_residue_st8) = curr_residue

    # classify st_2 it is always valid
    (st2_count, st2_res, st2_entropy) = count_in_interval(classify, test_set_x, ns_test_set_x_st2, curr_residue_st2, (start_frame/2-19), ns_test_set_x_st2.shape[0])
    # check if st5 is valid. if not return st2 count
    if (valid_st5 == 1):
        (st5_count, st5_res, st5_entropy) = count_in_interval(classify, test_set_x, ns_test_set_x_st5, curr_residue_st5, (start_frame/5-19), ns_test_set_x_st5.shape[0])
    else:
        st5_entropy = np.inf

    if (valid_st8 == 1):
        (st8_count, st8_res, st8_entropy) = count_in_interval(classify, test_set_x, ns_test_set_x_st8, curr_residue_st8, (start_frame/8-19), ns_test_set_x_st8.shape[0])
    else:
        st8_entropy = np.inf


    winner = np.nanargmin(np.array([st2_entropy, st5_entropy, st8_entropy]))

    if (winner == 0):
        # winner is stride 2
        return (global_count + st2_count)
    if (winner == 1):
        # winner is stride 5
        return (global_count + st5_count)
    if (winner == 2):
        # winner is stride 8
        return (global_count + st8_count)


def count_entire_movie(classify, test_set_x, data, valid, global_count, curr_residue, start_frame):

    (valid_st2,valid_st5,valid_st8) = valid
    (ns_test_set_x_st2,ns_test_set_x_st5,ns_test_set_x_st8) = data
    (curr_residue_st2, curr_residue_st5, curr_residue_st8) = curr_residue

    # classify st_2 it is always valid
    (st2_count, st2_res, st2_entropy) = count_in_interval(classify, test_set_x, ns_test_set_x_st2, curr_residue_st2, 0, ns_test_set_x_st2.shape[0])
    # check if st5 is valid. if not return st2 count
    if (valid_st5 == 1):
        (st5_count, st5_res, st5_entropy) = count_in_interval(classify, test_set_x, ns_test_set_x_st5, curr_residue_st5, 0, ns_test_set_x_st5.shape[0])
    else:
        st5_entropy = np.inf

    if (valid_st8 == 1):
        (st8_count, st8_res, st8_entropy) = count_in_interval(classify, test_set_x, ns_test_set_x_st8, curr_residue_st8, 0, ns_test_set_x_st8.shape[0])
    else:
        st8_entropy = np.inf

    winner = np.nanargmin(np.array([st2_entropy, st5_entropy, st8_entropy]))

    if (winner == 0):
        # winner is stride 2
        return (global_count + st2_count)
    if (winner == 1):
        # winner is stride 5
        return (global_count + st5_count)
    if (winner == 2):
        # winner is stride 8
        return (global_count + st8_count)



def test_benchmark_online(classify, test_set_x, batch_size):

    strides = (2,5,8)

    data_subset = "QUVACount-100" # YTSegments or QUVACount-100
    dataset_root = "../VideoCountingDataset/%s/" % data_subset
    vid_root = os.path.join(dataset_root, "video_slowdown/set3")
    #vid_root = os.path.join(dataset_root, "video")
    ann_root = os.path.join(dataset_root, "annotations")

    vid_files = glob.glob(os.path.join(vid_root, "*.avi"))

    cnt_gts_raw = pickle.load(open("vidGtData.p", "rb"))
    cnt_gts_original = np.zeros(100, dtype=np.int)
    cnt_gts_revised  = np.zeros(100, dtype=np.int)
    cnt_pred = np.zeros(100, dtype=np.int)


    for video_idx in range(100):

        vid_file = vid_files[video_idx]

        if data_subset == "YTsegments":
            matches = map(int, re.findall(r'\d+', vid_file))
            video_idx_from_filename = matches[-1]
            cnt_gts_original[video_idx] = cnt_gts_raw[video_idx_from_filename]

        vid_file_base, _ = os.path.splitext(os.path.basename(vid_file))
        ann_file = os.path.join(ann_root, "%s.npy" % vid_file_base)
        annotations = np.load(ann_file)

        cnt_gts_revised[video_idx] = len(annotations)

        print("VIDEO: %s" % vid_file_base)
        print("  video_file = %s" % vid_file)
        print("  ann_file   = %s" % ann_file)

        # Ground Truth count is number of count annotation locations
        cnt_gts_revised[video_idx] = len(annotations)
        print("  gt original = %i" % cnt_gts_original[video_idx])
        print("  gt revised  = %i" % cnt_gts_revised[video_idx])

        # load all 3 stride for this movie
        (data, valid) = load_movie_data(vid_file)
        #workaround for short movies
        if (data[0].shape[0]<81):
            global_count = count_entire_movie(classify, test_set_x, data, valid, 0, (0,0,0), 0)
            cnt_pred[video_idx] = global_count
            continue

        # get initial counting. all 3 stride for 200 frames.
        # i.e. st8 runs 25 times. st5 runs 40 times. st2 runs 100 times
        (global_count, curr_residue) = initial_count(classify, test_set_x, data, valid)

        # get the last multiple of 40 global frame
        numofiterations = get_inter_num(data,valid)
        for start_frame in range(200,200+(40*numofiterations),40):
            # from now on sync every 40 frames.
            # i.e. st8 runs 5 times. st5 8 times and st2 20 times.
            (global_count, curr_residue) = get_next_count(classify, test_set_x, data, valid, global_count, curr_residue, start_frame)

        # for frames that left get from each
        global_count = get_remain_count(classify, test_set_x, data, valid, global_count, curr_residue, 200+(40*numofiterations))
        cnt_pred[video_idx] = global_count

    if data_subset == "YTSegments":
        print("RESULTS ORIGINAL ANNOTATIONS")
        print_evaluation_summary(cnt_pred, cnt_gts_original)
        print_evaluation_summary_latex(cnt_pred, cnt_gts_original)

    # We now compute the evaluation metrics using cnt_pred and cnt_gts
    print("RESULTS REVISED OUR ANNOTATIONS")
    print_evaluation_summary(cnt_pred, cnt_gts_revised)
    print_evaluation_summary_latex(cnt_pred, cnt_gts_revised)