import warnings

import scipy
import os
from scipy.io import loadmat
import subprocess
import cv2
from PIL import Image, ImageDraw
import PIL.ImageFont as ImageFont
import numpy as np
import PIL.ImageColor as ImageColor
import pickle
import pandas as pd
import glob

try:
    # font = ImageFont.truetype('arial.ttf', 24)
    font = ImageFont.truetype("Tests/fonts/FreeMono.ttf", 84)
except IOError:
    font_path = os.path.join(cv2.__path__[0], 'qt', 'fonts', 'DejaVuSans.ttf')
    font = ImageFont.truetype(font_path, size=56)
color_space = [ImageColor.getrgb(n) for n, c in ImageColor.colormap.items()][7:] # avoid th aliceblue a light white one

min_iou_pascal_voc = 0.5
min_iou = min_iou_pascal_voc
"""
https://datasets.d2.mpi-inf.mpg.de/movieDescription/protected/grounded_characters/README.txt
= mat
The "mat" folder contains the bounding box annotations for a subset of characters in a few of training, validation and test movies. 
Each .mat file has the following structure:
- characters: the list of characters
- characters2frames: the list of video clip IDs & frame IDs, where the character was annotated
- characters2boxes: the list of corresponding bounding boxes

= frames
The "frames" folder contains the extracted video frames mentioned above, provided here for your convenience.

"""

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def plot_bbox_over_image(file_path, box, id_no_text, result_path):
    text_width, text_height = font.getsize('ID - 1')
    img = Image.open(file_path)
    img_draw = img.copy()
    draw = ImageDraw.Draw(img_draw)

    box[box < 0] = 0
    box[3] = [img.size[1] if box[3] > img.size[1] else box[3]][0]
    box[2] = [img.size[0] if box[2] > img.size[0] else box[2]][0]
    draw.rectangle(box.tolist(), width=10,
                   outline=color_space[id_no_text % len(color_space)])  # landmark plot
    margin = np.ceil(0.05 * text_height)
    draw.text(
        (box[0] + margin, box[3] - text_height - margin),
        str(id_no_text),
        fill='yellow',
        font=font)

    img_draw.save(os.path.join(result_path, 're-id_' + os.path.basename(file_path)))


def save_mdfs_to_jpg_from_clip(full_path, mdfs, save_path="/dataset/lsmdc/mdfs_of_20_clips/"):
    cap = cv2.VideoCapture(full_path)
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    for md in mdfs:
        if md > n_frames:
            warnings.warn("one of mdf exceeds No of frames !!!")
    vidcap = cv2.VideoCapture(full_path)
    success, image = vidcap.read()
    count = 0
    counted_mdfs = 0
    img_names = []
    mdfs = list(set(mdfs))  # remove duplicate mdfs
    if ".mp4" in full_path:
        full_path_modified = full_path.split("/")[-1].replace(".mp4", "")
    if ".avi" in full_path:
        full_path_modified = full_path.split("/")[-1].replace(".avi", "")
    while success:

        if count in mdfs and success:
            img_name = os.path.join(save_path, full_path_modified + '__' + str(count) + '.jpg')#f"{save_path}{full_path_modified}__{count}.jpg"
            img_names.append(img_name)
            cv2.imwrite(img_name, image)  # save frame as JPEG file
            print('Read a new frame: ', success)
            counted_mdfs += 1
        if counted_mdfs == len(mdfs):
            print("Done finding all MDFs")
            return img_names
        success, image = vidcap.read()
        count += 1
        # exit of infinite loop
        if count > 9999:
            return img_names

    return count

def main():
    save_mdfs_to_target = False
    annotation_path = '/media/mpii_reid/bbox/mat'
    data_set = 'training'

    video_db_path = '/mnt/share'
    # result_annotated_frames_mpii_path = '/home/hanoch/annotated_frames_mpii'
    result_annotated_frames_mpii_path = '/media/mpii_reid/bbox/frames'
    result_path = '/home/hanoch/results/face_reid/face_net/mpii'

    path = os.path.join(annotation_path, data_set)
    if not (os.path.isdir(path)):
        raise ValueError("{} Not mounted hence can not write to that folder ".format(path))

    mat_mov_filenames = [os.path.join(path, x) for x in os.listdir(path)
                 if x.endswith('mat')]

    if not bool(mat_mov_filenames):
        raise ValueError('No files at that folder')

    for file in mat_mov_filenames:
        # Locate movie clips folder
        video_name = os.path.basename(file).split('.mat')[0]
        print("Processing movie : ", video_name)
        result_path_movie = os.path.join(result_path, video_name)
        if not os.path.exists(result_path_movie):
            os.makedirs(result_path_movie)

        print("Video : ", video_name)
        # movie_full_path = subprocess.getoutput('find ' + video_db_path + ' -iname ' + '"*' + video_name + '*"')
        movie_full_path = subprocess.getoutput('find ' + video_db_path + ' -iname ' + '"*' + video_name + '"')
        if not movie_full_path:
            print("File not found", video_name)

        dest_path = os.path.join(result_annotated_frames_mpii_path, video_name)
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)


        annots = loadmat(file)
        print(annots)
        n_ids = annots['characters'].shape[0]
        print("no of characters {}", n_ids)
        for id in range(n_ids):# per id in movie
            print("Character name : {}".format(str(annots['characters'][id][0].item())))
            characters2frames_id = annots['characters2frames'][id, :]
            clip_name_for_id = characters2frames_id[0][0]
            frame_no_within_clip_name_for_id = characters2frames_id[0][1]
            bbox_id_all_clips_movie = annots['characters2boxes'][id, 0]
            assert (clip_name_for_id.shape[0] == bbox_id_all_clips_movie.shape[1])

            for ix , (clip, frame) in enumerate(zip(clip_name_for_id, frame_no_within_clip_name_for_id)):
                print(clip, frame)
                src_path_movie = os.path.join(movie_full_path, str(clip.astype(object)[0]) + '.avi')
                # dest_path_frame = dest_path + frame[0]
                if save_mdfs_to_target:
                    mdfs = [frame[0].astype('int') - 1]
                    save_mdfs_to_jpg_from_clip(full_path=src_path_movie, mdfs=mdfs, save_path=dest_path)
                else: # extract frames from MPII frames
                    full_path_modified = src_path_movie.split("/")[-1].replace(".avi", "")
                    # mdf_fname = os.path.join(dest_path, full_path_modified + '-' + str(str(frame[0])) + '.jpg')
                    frame_name = full_path_modified + '-' + str(str(frame[0]))
                    frame_full_path = subprocess.getoutput('find ' + dest_path + ' -iname ' + '"*' + frame_name + '*"')
                    if not frame_full_path:
                        print("File not found", video_name)
                    if '\n' in frame_full_path:
                        frame_full_path = frame_full_path.split('\n')[0]

                    bbox1 = bbox_id_all_clips_movie[:, ix].astype('float')
                    prop_bbox = np.array([bbox1[0] , bbox1[1], bbox1[0] +bbox1[2], bbox1[1] + bbox1[3]])
                    plot_bbox_over_image(file_path=frame_full_path, box=prop_bbox,
                                         id_no_text=id, result_path=result_path_movie)
                    # img = Image.open(mdf_fname)

            # subprocess.getoutput('cp -p ' + src_path_movie + ' ' + dest_path_frame)


def calc_precision_recall():
    data_sets = ['training', 'val', 'test']

    annotated_frames_mpii_path = '/media/mpii_reid/bbox/frames'
    # result_path = '/home/hanoch/results/face_reid/face_net/mpii'
    annotation_path = '/media/mpii_reid/bbox/mat'
    # detection_path = '/home/hanoch/results/face_reid/face_net'
    detection_path = '/media/results/face_reid'

    summary_list = list()
    for data_set in data_sets:
        path = os.path.join(annotation_path, data_set)

        if not (os.path.isdir(path)):
            raise ValueError("{} Not mounted hence can not write to that folder ".format(path))

        mat_mov_filenames = [os.path.join(path, x) for x in os.listdir(path)
                     if x.endswith('mat')]


        if not bool(mat_mov_filenames):
            raise ValueError('No files at that folder')

        for mat_file in mat_mov_filenames:
            # Locate movie clips folder
            video_name = os.path.basename(mat_file).split('.mat')[0]
            print("Processing movie : ", video_name)
            movie_pkl_path = os.path.join(detection_path, video_name)
            # '/1059_The_devil_wears_prada/method_dbscan_res_128_margin_40_eps_0.27_KNN_5'
            reid_dict_pkl_full_path_s = subprocess.getoutput('find ' + movie_pkl_path + ' -iname ' + '"*' + 're-id_res_' + '*"')
            if not reid_dict_pkl_full_path_s:
                print("File not found", reid_dict_pkl_full_path_s)
                continue
            # reid_dict_file = 're-id_res_128_0.95_eps_0.27_KNN_5.pkl'
            # pkl_path = os.path.join(detection_path, reid_dict_file)
            # Open detected/classified faces
            if '\n' in reid_dict_pkl_full_path_s:
                reid_dict_pkl_full_path_s = reid_dict_pkl_full_path_s.split('\n')
            else:
                reid_dict_pkl_full_path_s = [reid_dict_pkl_full_path_s]

            for reid_dict_pkl_full_path in reid_dict_pkl_full_path_s: # few results per movie
                with open(reid_dict_pkl_full_path, 'rb') as f:
                    mdf_face_id_all = pickle.load(f)
                print("ReId settings", os.path.basename(reid_dict_pkl_full_path))

                # csv load and get setup
                csv_path = os.path.dirname(reid_dict_pkl_full_path)
                try:
                    df_res = pd.read_csv(os.path.join(csv_path, 'setup.csv'), index_col=False)
                except Exception as e:
                    print(e)
                    continue
                new_header = df_res.iloc[0]  # grab the first row for the header
                df_res = df_res[1:]  # take the data less the header row
                df_res.columns = new_header

                annotation_converted_to_dict = dict()

                annots = loadmat(mat_file)
                n_ids = annots['characters'].shape[0]
                print("no of characters {}", n_ids)
                # RECALL : Percentage of GT that appears in frames/MDFs: Go over the GT list find the IOU>min_iou_pascal_voc over classified faces bbox
                # + make sure all classification belongs to the same ID otherwise it is just detection and multiple IDs (hard positives) will not be detected
                false_negatives = 0
                true_positives = 0
                detected_but_not_classified = 0
                id_annotations = 0
                gt_id_to_detected_id_map = dict()
                for id in range(n_ids):# per id in movie over all clips
                    detected_id = list()
                    print("Character name : {} No #{}".format(str(annots['characters'][id][0].item()), id))
                    characters2frames_id = annots['characters2frames'][id, :]
                    clip_name_for_id = characters2frames_id[0][0]
                    if clip_name_for_id.size == 0:
                        print("video_name : missing Id annotations ", video_name)
                        continue
                    frame_no_within_clip_name_for_id = characters2frames_id[0][1]
                    bbox_id_all_clips_movie = annots['characters2boxes'][id, 0]
                    id_annotations += clip_name_for_id.shape[0]
                    assert (clip_name_for_id.shape[0] == bbox_id_all_clips_movie.shape[1])

                    for ix, (clip, frame) in enumerate(zip(clip_name_for_id, frame_no_within_clip_name_for_id)):
                        # print(clip, frame)
                        bbox1 = bbox_id_all_clips_movie[:, ix].astype('float')
                        prop_bbox = np.array([bbox1[0], bbox1[1], bbox1[0] +bbox1[2], bbox1[1] + bbox1[3]])

                        frame_fname = str(clip.astype(object)[0]) + '-' + str(frame[0]) + '.jpg'

                        annotation_converted_fname = annotation_converted_to_dict.get(frame_fname, None)
                        if annotation_converted_fname:
                            annotation_converted_fname.update({id: prop_bbox})
                            annotation_converted_to_dict[frame_fname] = annotation_converted_fname
                        else:
                            tmp_d = {id: prop_bbox}
                            annotation_converted_to_dict.update({frame_fname: tmp_d})

                        # Find frame in detections
                        frame_dict_value = mdf_face_id_all.get(frame_fname, -1)
                        if frame_dict_value == -1:
                            false_negatives += 1 # since GT has annotations in this frame out of specific clip while no detections
                        else: # Detections were made over that frame/clip
                            max_iou = 0
                            det_id_in_frame = False
                            for det_ids in list(frame_dict_value.values()):
                                iou = bb_intersection_over_union(det_ids['bbox'], prop_bbox)
                                if iou > max_iou:
                                    max_iou = iou
                                    if iou>min_iou_pascal_voc:
                                        if det_ids['id'] == -1:
                                            false_negatives += 1
                                            detected_but_not_classified += 1
                                            print("Id detected but not classified ", frame_fname,
                                                  annots['characters'][id][0].item(0))
                                        else:
                                            true_positives += 1
                                            det_id_in_frame = True
                                            detected_id.append(det_ids['id'])
                            if not det_id_in_frame:
                                false_negatives += 1
                    # Validate mapping of dbscan Ids to GT annotations
                    uniqu_id, c = np.unique(detected_id, return_counts=True)
                    if uniqu_id.size > 0:
                        gt_id_to_detected_id_map.update({uniqu_id[np.argmax(c)]: id}) # in case substitution of id or replicated
                    if uniqu_id.shape[0]>1:
                        warnings.warn("Substitution of IDs :{} times: {}".format(uniqu_id, c))

                recall = true_positives/(true_positives + false_negatives)
                # now that we have ,mapping gt_id_to_detected_id_map compute precision
                # FALSE POSITIVE CALC
                false_positive = 0
                for frame_number, v in mdf_face_id_all.items():
                    frame_dict_value = mdf_face_id_all.get(frame_number, None)
                    if not frame_dict_value: # no face detection at that MDF
                        continue
                    for det_ids in list(frame_dict_value.values()):
                        if det_ids['id'] != -1:
                            for id_k, v_bbox in annotation_converted_to_dict[frame_number].items():# run over all GT annotations
                                iou = bb_intersection_over_union(det_ids['bbox'], v_bbox)
                                if iou > min_iou_pascal_voc:
                                    map_det_to_gt = gt_id_to_detected_id_map.get(det_ids['id'], None)# overlapping of GT and prediction means but with different IDs means FP of detected ID with other face ID
                                    if map_det_to_gt:
                                        if map_det_to_gt != id_k:
                                            false_positive += 1

                if false_positive + true_positives == 0:
                    precision = 0
                else:
                    precision = true_positives/(false_positive + true_positives)

                res_dict = dict()
                res_dict.update({'tp': true_positives, 'fn': false_negatives, 'recall': recall, 'precision': precision, 'detected_but_not_classified ratio': detected_but_not_classified/(id_annotations)})
                res_dict.update({'recluster_hard_positives': df_res.recluster_hard_positives})
                res_dict.update({'mtcnn_margin': df_res.mtcnn_margin.item(), 'min_face_res': df_res.min_face_res.item(), 'dbscan_eps' : df_res.cluster_threshold.item()})
                res_dict.update({"min_cluster_size": df_res.min_cluster_size.item(), 'reid_method': df_res.reid_method.item(), 'min_cluster_size': df_res.min_cluster_size.item()})
                res_dict.update({'video_name': video_name})

                summary_list.append(res_dict)
                print("ID {} no {} true_positives {}, false_negatives {} recall {} detected_but_not_classified ratio {} FP: {} Precision {}"
                      .format(annots['characters'][id][0].item(0), id, true_positives, false_negatives, recall, detected_but_not_classified/(id_annotations), false_positive, precision))
            # PRECISION :
    df = pd.DataFrame(summary_list)
    df.to_csv(os.path.join(detection_path, 'precision_recall.csv'), index=False)
    return

if __name__ == '__main__':
    if 1:
        calc_precision_recall()
    else:
        main()
