'''
live repetition counting system
Ofir Levy, Lior Wolf
Tel Aviv University
'''
import os
import re
import numpy as np
import glob
import cv2
import pickle
from common import *

from bluevelvet.utils.evaluation import *


def test_benchmark_offline(classify, test_set_x, batch_size):

    strides = (2,5,8)

    data_subset = "YTSegments"  #QUVACount-100"
    dataset_root = "../VideoCountingDataset/%s/" % data_subset
    vid_root = os.path.join(dataset_root, "video_slowdown/set1")
    ann_root = os.path.join(dataset_root, "annotations")

    vid_files = glob.glob(os.path.join(vid_root, "*.avi"))

    cnt_gts_raw = pickle.load(open("vidGtData.p", "rb"))
    cnt_gts_original = np.zeros(100, dtype=np.int)
    cnt_gts_revised  = np.zeros(100, dtype=np.int)
    cnt_pred = np.zeros(100, dtype=np.int)

    num_videos = 100
    countArr = np.zeros(shape=(num_videos, len(strides)))
    entropy = np.zeros(shape=(num_videos, len(strides)))
    num_entropy = np.zeros(shape=(num_videos, len(strides)))

    for stride_idx, stride in enumerate(strides):
        for video_idx in range(num_videos):

            vid_file = vid_files[video_idx]

            if data_subset == "YTsegments":
                matches = map(int, re.findall(r'\d+', vid_file))
                video_idx_from_filename = matches[-1]
                cnt_gts_original[video_idx] = cnt_gts_raw[video_idx_from_filename]

            vid_file_base, _ = os.path.splitext(os.path.basename(vid_file))
            ann_file = os.path.join(ann_root, "%s.npy" % vid_file_base)
            annotations = np.load(ann_file)

            cnt_gts_revised[video_idx] = len(annotations)

            print("VIDEO: YT_SEG_%i" % video_idx)
            print("  stride      = %i" % stride)
            print("  video_file  = %s" % vid_file)
            print("  ann_file    = %s" % ann_file)
            if data_subset == "YTSegments":
                print("  gt original = %i" % cnt_gts_original[video_idx])
            print("  gt revised  = %i" % cnt_gts_revised[video_idx])

            mydatasets = load_next_test_data(vid_file, stride)
            ns_test_set_x, valid_ds = mydatasets
            if (valid_ds == 0):  # file not axists
                continue

            test_set_x.set_value(ns_test_set_x, borrow=True)
            n_samples = ns_test_set_x.shape[0]

            out_list = [classify(i) for i in range(n_samples)]

            frame_counter = 0
            rep_counter = 0
            curr_entropy = 0
            ent_cnt = 0

            for batch_num in range(len(out_list)):

                output_label_batch , pYgivenX = out_list[batch_num]

                # Removed index in following line
                output_label = output_label_batch[0] + 3 # moving from label to cycle length
                pYgivenX[pYgivenX==0] = np.float32(1e-30) # hack to output valid entropy

                # calc entropy
                curr_entropy = curr_entropy - (pYgivenX*np.log(pYgivenX)).sum()
                ent_cnt= ent_cnt + 1
                # integrate counting
                if (batch_num==0):
                    rep_counter = 20 / (output_label)
                    frame_counter = 20 % (output_label)
                else:
                    frame_counter += 1
                    if (frame_counter >= output_label):
                        rep_counter += 1
                        frame_counter = 0

            countArr[video_idx, stride_idx] = rep_counter
            entropy[video_idx, stride_idx] = curr_entropy
            num_entropy[video_idx, stride_idx] = ent_cnt

    if data_subset == "YTSegments":
        print("BEST STRIDE (original annotations):")
        absdiff_o = abs(countArr-np.expand_dims(cnt_gts_original[0:num_videos], -1))
        best_strides = np.argmin(absdiff_o, axis=1)
        cnt_pred_best_stride = countArr[np.arange(len(countArr)),best_strides]
        print(cnt_pred_best_stride.shape)
        print_evaluation_summary(cnt_pred_best_stride, cnt_gts_original[0:num_videos])
        print_evaluation_summary_latex(cnt_pred_best_stride, cnt_gts_original[0:num_videos])
        print("#"*60)

    print("BEST STRIDE (revised annotations):")
    absdiff_o = abs(countArr-np.expand_dims(cnt_gts_revised[0:num_videos], -1))
    best_strides = np.argmin(absdiff_o, axis=1)
    cnt_pred_best_stride = countArr[np.arange(len(countArr)),best_strides]
    print_evaluation_summary(cnt_pred_best_stride, cnt_gts_revised[0:num_videos])
    print_evaluation_summary_latex(cnt_pred_best_stride, cnt_gts_revised[0:num_videos])
    print("#"*60)

    #min_err_cnt_o = absdiff_o.min(axis=1)
    #min_err_perc_o = min_err_cnt_o/gt[:,1]
    #err_perc_o = np.average(min_err_perc_o)*100
    #print 'alpha = 1: precentage error:    %.2f%%' % (err_perc_o)

    # Compute the median count for the strides
    cnt_pred_median = np.median(countArr,axis=1)

    if data_subset == "YTSegments":
        print("MEDIAN STRIDE (original annotations):")
        print_evaluation_summary(cnt_pred_median, cnt_gts_original[0:num_videos])
        print_evaluation_summary_latex(cnt_pred_median, cnt_gts_original[0:num_videos])
        print("#"*60)

    print("MEDIAN STRIDE (revised annotations):")
    print_evaluation_summary(cnt_pred_median, cnt_gts_revised[0:num_videos])
    print_evaluation_summary_latex(cnt_pred_median, cnt_gts_revised[0:num_videos])
    print("#"*60)

    # TODO: choose based on lowest entropy

    xx = entropy/num_entropy
    chosen_stride = np.nanargmin(xx,axis=1)
    m = np.arange(chosen_stride.shape[0])*len(strides)
    m = m + chosen_stride
    flt = countArr.flatten()
    cnt_pred_entropy = flt[m]

    if data_subset == "YTSegments":
        print("ENTROPY STRIDE (original annotations):")
        print_evaluation_summary(cnt_pred_entropy, cnt_gts_original[0:num_videos])
        print_evaluation_summary_latex(cnt_pred_entropy, cnt_gts_original[0:num_videos])
        diff = cnt_pred_entropy - cnt_gts_original[0:num_videos]
        np.save("./count_differences_entropy.npy", diff)
        print("#"*60)

    print("ENTROPY STRIDE (revised annotations):")
    print_evaluation_summary(cnt_pred_entropy, cnt_gts_revised[0:num_videos])
    print_evaluation_summary_latex(cnt_pred_entropy, cnt_gts_revised[0:num_videos])
    print("#"*60)
