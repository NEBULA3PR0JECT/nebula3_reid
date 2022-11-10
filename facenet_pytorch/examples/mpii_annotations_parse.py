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

try:
    # font = ImageFont.truetype('arial.ttf', 24)
    font = ImageFont.truetype("Tests/fonts/FreeMono.ttf", 84)
except IOError:
    font_path = os.path.join(cv2.__path__[0], 'qt', 'fonts', 'DejaVuSans.ttf')
    font = ImageFont.truetype(font_path, size=56)
color_space = [ImageColor.getrgb(n) for n, c in ImageColor.colormap.items()][7:] # avoid th aliceblue a light white one


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

    frame_no_within_clip_name_for_id = characters2frames_id[0][1]
    for clip, frame in zip(clip_name_for_id, frame_no_within_clip_name_for_id):
        print(clip, frame)

