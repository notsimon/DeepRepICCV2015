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

import matplotlib.pyplot as plt
from bluevelvet.utils.evaluation import *
from bluevelvet.plotting.colors import *
from bluevelvet.plotting.plotting import *

init_plotting()
sns.set_context("paper")
colors = tableau_colors()


def get_inter_num(data, valid):
    (ns_test_set_x_st2,ns_test_set_x_st5,ns_test_set_x_st8) = data
    (valid_st2,valid_st5,valid_st8) = valid
    if (valid_st8 == 1):
        return ((ns_test_set_x_st8.shape[0]-6)/5)
    if (valid_st5 == 1):
        return ((ns_test_set_x_st8.shape[0]-21)/8)
    else:
        return ((ns_test_set_x_st8.shape[0]-81)/20)


def load_movie_data(fileName, search_bounding_box=True):
    (ns_test_set_x_st2, valid_st2) = load_next_test_data(fileName, 2, search_bounding_box=search_bounding_box)
    (ns_test_set_x_st5, valid_st5) = load_next_test_data(fileName, 5, search_bounding_box=search_bounding_box)
    (ns_test_set_x_st8, valid_st8) = load_next_test_data(fileName, 8, search_bounding_box=search_bounding_box)

    return ((ns_test_set_x_st2,ns_test_set_x_st5,ns_test_set_x_st8), (valid_st2,valid_st5,valid_st8))


def count_in_interval(classify, test_set_x, ns_test_set_x, frame_residue, start, end):

    assert start <= end
    if start == end:
        return 0, 0, 0, None

    test_set_x.set_value(ns_test_set_x, borrow=True)

    rep_counts = 0
    entropies = []

    for i in range(start, end):
        output_label , pYgivenX = classify(i)
        pYgivenX[pYgivenX==0] = np.float32(1e-30) # hack to output valid entropy

        # Compute the entropy
        entropy = (pYgivenX*np.log(pYgivenX)).sum()
        entropies.append(entropy)

        # Moving from label to cycle length
        output_label += 3

        if i == 0:
            rep_counts = 20 / int(output_label)
            frame_residue = 20 % int(output_label)
        else:
            frame_residue += 1
            if frame_residue >= output_label:
                rep_counts += 1
                frame_residue = 0

    # Compute the average entropy
    entropies = np.asarray(entropies, dtype=np.float32)
    avg_entropy = np.mean(entropies)

    return rep_counts, frame_residue, avg_entropy, entropies



def initial_count(classify, test_set_x, data, valid):

    (valid_st2,valid_st5,valid_st8) = valid
    (ns_test_set_x_st2,ns_test_set_x_st5,ns_test_set_x_st8) = data

    # Stride 2 (=always valid)
    st2_count, st2_res, st2_avg_entropy, st2_entropies = count_in_interval(classify, test_set_x, ns_test_set_x_st2, 0, 0, 81)  #100 - 19 etc.

    st5_avg_entropy, st5_entropies = np.inf, None
    st8_avg_entropy, st8_entropies = np.inf, None

    # Stride 5
    if valid_st5 == 1:
        st5_count, st5_res, st5_avg_entropy, st5_entropies = count_in_interval(classify, test_set_x, ns_test_set_x_st5, 0, 0, 21)

    # Stride 8
    if valid_st8 == 1:
        st8_count, st8_res, st8_avg_entropy, st8_entropies = count_in_interval(classify, test_set_x, ns_test_set_x_st8, 0, 0, 6)

    winner_stride_index = np.nanargmin(np.array([st2_avg_entropy, st5_avg_entropy, st8_avg_entropy]))
    entropies = (st2_entropies, st5_entropies, st8_entropies)

    if winner_stride_index == 0:
        # Winner is stride 2
        res_frames_stride = (st2_res*2/2,st2_res*2/5, st2_res*2/8)
        current_count = st2_count
    elif winner_stride_index == 1:
        # Winner is stride 5
        res_frames_stride = (st5_res*5/2,st5_res*5/5, st5_res*5/8)
        current_count = st5_count
    else:
        # Winner is stride 8
        res_frames_stride = (st8_res*8/2,st8_res*8/5, st8_res*8/8)
        current_count = st8_count

    return current_count, res_frames_stride, entropies, winner_stride_index


def get_next_count(classify, test_set_x, data, valid, global_count, curr_residue, start_frame):

    (valid_st2,valid_st5,valid_st8) = valid
    (ns_test_set_x_st2,ns_test_set_x_st5,ns_test_set_x_st8) = data
    (curr_residue_st2, curr_residue_st5, curr_residue_st8) = curr_residue

    # Stride 2 (=always valid)
    st2_count, st2_res, st2_avg_entropy, st2_entropies = \
        count_in_interval(classify, test_set_x, ns_test_set_x_st2, curr_residue_st2, (start_frame/2-19), (start_frame/2-19)+20)

    st5_avg_entropy, st5_entropies, st5_res, st5_count = np.inf, None, None, None
    st8_avg_entropy, st8_entropies, st8_res, st8_count = np.inf, None, None, None

    # Stride 5
    if valid_st5 == 1:
        st5_count, st5_res, st5_avg_entropy, st5_entropies = \
            count_in_interval(classify, test_set_x, ns_test_set_x_st5, curr_residue_st5, (start_frame/5-19), (start_frame/5-19)+8)

    # Stride 8
    if valid_st8 == 1:
        st8_count, st8_res, st8_avg_entropy, st8_entropies = \
            count_in_interval(classify, test_set_x, ns_test_set_x_st8, curr_residue_st8, (start_frame/8-19), (start_frame/8-19)+5)

    winner_stride_index = np.nanargmin(np.array([st2_avg_entropy, st5_avg_entropy, st8_avg_entropy]))
    entropies = (st2_entropies, st5_entropies, st8_entropies)
    current_count = global_count

    if winner_stride_index == 0:
        # Winner is stride 2
        res_frames_stride = (st2_res*2/2,st2_res*2/5, st2_res*2/8)
        current_count += st2_count

    elif winner_stride_index == 1:
        # Winner is stride 5
        res_frames_stride = (st5_res*5/2,st5_res*5/5, st5_res*5/8)
        current_count += st5_count
    else:
        # Winner is stride 8
        res_frames_stride = (st8_res*8/2,st8_res*8/5, st8_res*8/8)
        current_count += st8_count

    return current_count, res_frames_stride, entropies, winner_stride_index


def get_remain_count(classify, test_set_x, data, valid, global_count, curr_residue, start_frame):

    (valid_st2,valid_st5,valid_st8) = valid
    (ns_test_set_x_st2,ns_test_set_x_st5,ns_test_set_x_st8) = data
    (curr_residue_st2, curr_residue_st5, curr_residue_st8) = curr_residue

    # Stride 2 (=always valid)
    st2_count, st2_res, st2_avg_entropy, st2_entropies = \
        count_in_interval(classify, test_set_x, ns_test_set_x_st2, curr_residue_st2, (start_frame/2-19), ns_test_set_x_st2.shape[0])

    st5_avg_entropy, st5_entropies, st5_res, st5_count = np.inf, None, None, None
    st8_avg_entropy, st8_entropies, st8_res, st8_count = np.inf, None, None, None

    # Stride 5
    if valid_st5 == 1:
        st5_count, st5_res, st5_avg_entropy, st5_entropies = \
            count_in_interval(classify, test_set_x, ns_test_set_x_st5, curr_residue_st5, (start_frame/5-19), ns_test_set_x_st5.shape[0])

    # Stride 8
    if valid_st8 == 1:
        st8_count, st8_res, st8_avg_entropy, st8_entropies = \
            count_in_interval(classify, test_set_x, ns_test_set_x_st8, curr_residue_st8, (start_frame/8-19), ns_test_set_x_st8.shape[0])

    winner_stride_index = np.nanargmin(np.array([st2_avg_entropy, st5_avg_entropy, st8_avg_entropy]))
    entropies = (st2_entropies, st5_entropies, st8_entropies)
    current_count = global_count

    if winner_stride_index == 0:
        # Winner is stride 2
        current_count += st2_count
    elif winner_stride_index == 1:
        # Winner is stride 5
        current_count += st5_count
    else:
        # Winner is stride 8
        current_count += st8_count

    return current_count, entropies, winner_stride_index


def count_entire_movie(classify, test_set_x, data, valid, global_count, curr_residue, start_frame):
    '''
    This method is only used for short movies, for which we count without
    splitting into 'start_count', 'remain_count' and 'final_count'
    '''

    (valid_st2,valid_st5,valid_st8) = valid
    (ns_test_set_x_st2,ns_test_set_x_st5,ns_test_set_x_st8) = data
    (curr_residue_st2, curr_residue_st5, curr_residue_st8) = curr_residue

    # Stride 2 (=always valid)
    st2_count, st2_res, st2_avg_entropy, st2_entropies = \
        count_in_interval(classify, test_set_x, ns_test_set_x_st2, curr_residue_st2, 0, ns_test_set_x_st2.shape[0])

    st5_avg_entropy, st5_entropies, st5_res, st5_count = np.inf, None, None, None
    st8_avg_entropy, st8_entropies, st8_res, st8_count = np.inf, None, None, None

    # Stride 5
    if valid_st5 == 1:
        st5_count, st5_res, st5_avg_entropy, st5_entropies = \
            count_in_interval(classify, test_set_x, ns_test_set_x_st5, curr_residue_st5, 0, ns_test_set_x_st5.shape[0])

    # Stride 8
    if valid_st8 == 1:
        st8_count, st8_res, st8_avg_entropy, st8_entropies = \
            count_in_interval(classify, test_set_x, ns_test_set_x_st8, curr_residue_st8, 0, ns_test_set_x_st8.shape[0])

    winner_stride_index = np.nanargmin(np.array([st2_avg_entropy, st5_avg_entropy, st8_avg_entropy]))
    entropies = (st2_entropies, st5_entropies, st8_entropies)
    current_count = global_count

    if winner_stride_index == 0:
        # Winner is stride 2
        current_count += st2_count
    elif winner_stride_index == 1:
        # Winner is stride 5
        current_count += st5_count
    else:
        # Winner is stride 8
        current_count += st8_count

    return current_count, entropies, winner_stride_index


def analyze_online_counting(classify, test_set_x, batch_size):

    strides = (2,5,8)
    search_bounding_box = False

    dataset_root = "/home/trunia1/data/VideoCountingDatasetClean/LevyWolf_Segment/videos/"

    # Path containing avi files
    vid_root = os.path.join(dataset_root, "video")
    vid_files = glob.glob(os.path.join(vid_root, "*.avi"))
    vid_files.sort()

    # Wolf' and Our annotations
    wolf_ann_file = os.path.join(dataset_root, "vidGtData.p")
    our_ann_root  = os.path.join(dataset_root, "annotations/cycle_annotations")

    cnt_gts_raw = pickle.load(open(wolf_ann_file, "rb"))
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
        ann_file = os.path.join(our_ann_root, "%s.npy" % vid_file_base)
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
        (data, valid) = load_movie_data(vid_file, search_bounding_box)

        # Save all entropies
        entropies = []
        winner_strides = []
        avg_entropies = []

        def compute_avg_entropies(ent_all):
            ent_all_np = np.asarray(ent_all)
            avg_ent = np.zeros(3, np.float32)
            for i in range(3):
                if ent_all_np[i] is None:
                    avg_ent[i] = 0
                else:
                    avg_ent[i] = np.mean(ent_all_np[i])
            return avg_ent

        if data[0].shape[0] < 81:
            # Workaround for short movies
            global_count, entropies_current, win_stride_cur = count_entire_movie(classify, test_set_x, data, valid, 0, (0,0,0), 0)
            cnt_pred[video_idx] = global_count
            entropies.append(entropies_current)
            winner_strides.append(strides[win_stride_cur])
            avg_entropies.append(compute_avg_entropies(entropies_current))
        else:
            # Longer movies
            # get initial counting. all 3 stride for 200 frames.
            # i.e. st8 runs 25 times. st5 runs 40 times. st2 runs 100 times
            global_count, curr_residue, entropies_current, win_stride_cur = initial_count(classify, test_set_x, data, valid)

            entropies.append(entropies_current)
            winner_strides.append(strides[win_stride_cur])
            avg_entropies.append(compute_avg_entropies(entropies_current))

            # Get the last multiple of 40 global frame
            numofiterations = get_inter_num(data,valid)

            for start_frame in range(200, 200+(40*numofiterations),40):
                # from now on sync every 40 frames.
                # i.e. st8 runs 5 times. st5 8 times and st2 20 times.
                global_count, curr_residue, entropies_current, win_stride_cur = get_next_count(classify, test_set_x, data, valid, global_count, curr_residue, start_frame)
                winner_strides.append(strides[win_stride_cur])
                avg_entropies.append(compute_avg_entropies(entropies_current))

                #print("Entropies, Start Frame = {}".format(start_frame))
                #for j in range(3):
                #    print("  Stride {}. Num Entropies: {}. Entropies: {}".format(strides[j], len(entropies_current[j]), str(entropies_current[j])))

            # for frames that left get from each
            final_offset = 200+(40*numofiterations)
            global_count, entropies_current, win_stride_cur = get_remain_count(classify, test_set_x, data, valid, global_count, curr_residue, final_offset)
            entropies.append(entropies_current)
            winner_strides.append(strides[win_stride_cur])

            cnt_pred[video_idx] = global_count

        ########################################################################
        ########################################################################
        ########################################################################

        # CODE FOR PLOTTING MEAN ENTROPIES
        f, ax = plt.subplots(ncols=2, nrows=1, figsize=(8,4))

        avg_entropies_np = np.asarray(avg_entropies)

        for j in range(3):
            ax[0].plot(avg_entropies_np[:,j], label="Stride {}".format(strides[j]))
            ax[0].set_xlabel("Synchronization Time")
            ax[0].set_ylabel("Entropy")

        ax[0].set_title("Average Entropies")
        ax[0].set_xlim([0, len(avg_entropies_np)])
        legend = ax[0].legend()

        np_winners = np.asarray(winner_strides)
        ax[1].plot(np_winners)
        ax[1].set_title("Most Probable Stride (Lowest Entropy)")
        ax[1].set_xlabel("Synchronization Time")
        ax[1].set_ylabel("Winner Stride")
        ax[1].set_xlim([0, len(avg_entropies_np)])
        ax[1].set_yticks([2,5,8])
        ax[1].set_ylim([0, 10])

        plt.suptitle("Average Entropy (YT_Seg_{})".format(video_idx))

        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        #plt.show()
        plot_out = os.path.join(entropy_plot_dir, "YT_Seg_{}.pdf".format(video_idx))
        plt.savefig(plot_out)
        plt.close()


        # CODE FOR PLOTTING ALL ENTROPIES
        # f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        # ax = [ax1, ax2, ax3, ax4]
        #
        # # Plot the entropies
        # for j in range(3):
        #
        #     entropies_stride = np.empty(0)
        #     for k in range(len(entropies)):
        #         entropies_stride = np.append(entropies_stride, entropies[k][j])
        #
        #     ax[j].plot(entropies_stride, label="Stride {}".format(strides[j]), c=colors[j])
        #     ax[j].set_title("Stride {}".format(strides[j]))
        #     ax[j].set_xlabel("Block Index")
        #     ax[j].set_ylabel("Entropy")
        #
        # np_winners = np.asarray(winner_strides)
        # ax[3].plot(np_winners)
        # ax[3].set_xlabel("Sync Time")
        # ax[3].set_ylabel("Winner Stride")
        #
        # plt.tight_layout()
        #
        # #plt.suptitle("Entropies (YT_Seg_{})".format(video_idx))
        # plt.show()


    if data_subset == "YTSegments":
        print("RESULTS ORIGINAL ANNOTATIONS")
        print_evaluation_summary(cnt_pred, cnt_gts_original)
        print_evaluation_summary_latex(cnt_pred, cnt_gts_original)

    # We now compute the evaluation metrics using cnt_pred and cnt_gts
    print("RESULTS REVISED OUR ANNOTATIONS")
    print_evaluation_summary(cnt_pred, cnt_gts_revised)
    print_evaluation_summary_latex(cnt_pred, cnt_gts_revised)