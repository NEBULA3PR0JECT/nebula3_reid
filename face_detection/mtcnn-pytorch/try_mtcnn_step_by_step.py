# export PYTHONPATH=/notebooks/pip_install
from src import detect_faces, show_bboxes
from PIL import Image
import sys
import os
import numpy as np
import tqdm
# Constants
min_face_res = 64


# folders
curr_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(curr_dir)
sys.path.append('/home/hanoch/notebooks')
# sys.path.append('/home/notebooks/nebula3_reid/face_detection/mtcnn-pytorch/src/weights')
# img = Image.open('/notebooks/nebula3_reid/face_detection/mtcnn-pytorch/images/office1.jpg')#('images/office1.jpg')

path_mdf = '/home/hanoch/mdfs2_lsmdc' #'/notebooks/nebula3_reid/mdfs2_lsmdc' #'/notebooks/nebula3_reid/reults_dir/msr_vtt/mdf/videos'#'/notebooks/nebula3_videoprocessing/videoprocessing/msr_vtt/mdf/videos'
result_dir = '/home/hanoch/results/face_reid/mdf_lsmdc1'
# result_dir = os.path.join(path_mdf, 'face_det')
plot_conf=True

if not os.path.exists(result_dir):
    os.makedirs(result_dir)

filenames = [os.path.join(path_mdf, x) for x in os.listdir(path_mdf)
                    if x.endswith('png') or x.endswith('jpg')]
if filenames is None:
    raise ValueError('No files at that folder')
for file in tqdm.tqdm(filenames):
    img = Image.open(file)#('images/office1.jpg')
# Image pyramid followed by 3 stages : min_face_size=20.0 ; image is resized 
    bounding_boxes, landmarks = detect_faces(img)  # thresholds=[0.6, 0.7, 0.8]  , nms_thresholds=[0.7, 0.7, 0.7]
    # threshold: a float number, threshold on the probability of a face when generating  bounding boxes from predictions of the net.
    if isinstance(bounding_boxes, np.ndarray) and bounding_boxes.any() or isinstance(bounding_boxes, list) and bool(bounding_boxes):
        if [(bounding_boxes[2] - bounding_boxes[0])>min_face_res and (bounding_boxes[3] - bounding_boxes[1])>min_face_res for bounding_boxes in bounding_boxes][0]:
            print('This face resolution {} too low {}'.format(bounding_boxes, min_face_res))
        resolution_color = list()
        for b in bounding_boxes:
            print("confidence{}".format(str(b[-1].__format__('.2f'))))
            if ((b[2] - b[0])>min_face_res and (b[3] - b[1])>min_face_res):
                resolution_color.append('white')
            else:
                resolution_color.append('red')

    img_copy = show_bboxes(img, bounding_boxes, landmarks, plot_conf=plot_conf, resolution_color=resolution_color)

    # img_copy.save(os.path.join(curr_dir, 'image_out.png'))
    
    img_copy.save(os.path.join(result_dir, os.path.basename(file)))
