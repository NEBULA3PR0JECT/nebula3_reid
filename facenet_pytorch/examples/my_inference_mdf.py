import sys
import os

curr_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(curr_dir)
sys.path.append('/home/hanoch/notebooks/nebula3_reid')
sys.path.append('/home/hanoch/notebooks/nebula3_reid/facenet_pytorch')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
from PIL import Image, ImageDraw
import cv2

from facenet_pytorch import MTCNN, InceptionResnetV1, extract_face, fixed_image_standardization
# from facenet_pytorch.models import mtcnn, inception_resnet_v1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

rel_path = 'nebula3_reid/facenet_pytorch'
result_path = '/home/hanoch/results/face_reid/face_net'
path_mdf = '/home/hanoch/mdfs2_lsmdc'

# plt.savefig('/home/hanoch/notebooks/nebula3_reid/face_tens.png')


workers = 0 if os.name == 'nt' else 4
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))
#### Define MTCNN module
"""
Default params shown for illustration, but not needed. Note that, since MTCNN is a collection of neural nets and other code, the device must be passed in the following way to enable copying of objects when needed internally.

See `help(MTCNN)` for more details.

"""
plot_cropped_faces = True
detection_with_landmark = True
if plot_cropped_faces:
    print("FaceNet output is an image not an embedding")

keep_all = True
min_face_res = 64
save_images = True
# post_process=True => fixed_image_standardization 
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True, keep_all=keep_all, 
    device=device ) #post_process=False
# Modify model to VGGFace based and resnet
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def collate_fn(x):
    return x[0]

# dataset = datasets.ImageFolder(os.path.join(rel_path, 'data/test_images'))
# dataset = datasets.ImageFolder(path_mdf)
# dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
# loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)


aligned = []
names = []

filenames = [os.path.join(path_mdf, x) for x in os.listdir(path_mdf)
                    if x.endswith('png') or x.endswith('jpg')]
if filenames is None:
    raise ValueError('No files at that folder')

for crop_inx, file in enumerate(tqdm.tqdm(filenames)):
    img = Image.open(file)#('images/office1.jpg')

    if detection_with_landmark:
        boxes, probs, points = mtcnn.detect(img, landmarks=True)
        img_draw = img.copy()
        if boxes is not None:
        # Draw boxes and save faces
            draw = ImageDraw.Draw(img_draw)
            for i, (box, point) in enumerate(zip(boxes, points)):

                print("confidence: {}".format(str(probs[0].__format__('.2f'))))
                if (box[2] - box[0])>min_face_res and (box[3] - box[1])>min_face_res:
                    face_bb_resolution = 'res_ok'
                else:
                    face_bb_resolution = 'res_bad'


                draw.rectangle(box.tolist(), width=5)
                for p in point:
                    # draw.rectangle((p - 10).tolist() + (p + 10).tolist(), width=4)
                    draw.rectangle((p - 1).tolist() + (p + 1).tolist(), width=2)

                    # for i in range(5):
                    #     draw.ellipse([
                    #         (p[i] - 1.0, p[i + 5] - 1.0),
                    #         (p[i] + 1.0, p[i + 5] + 1.0)
                    #     ], outline='blue')

                if save_images:
                    save_path = os.path.join(result_path, str(crop_inx) + '_' + face_bb_resolution + '_' + str(point.shape[0]) + '_face_{}'.format(i) + os.path.basename(file))
                else:
                    save_path = None

                print("N-Landmarks {}".format(point.shape[0]))
                names.append(str(crop_inx) + '_' + face_bb_resolution + '_'+ 'face_{}'.format(i) + os.path.basename(file))

                x_aligned = extract_face(img, box, save_path=save_path)
                x_aligned = fixed_image_standardization(x_aligned) # as done by the mtcnn.forward() to inject to FaceNet
                aligned.append(x_aligned)
            if save_images:
                img_draw.save(os.path.join(result_path, str(crop_inx) + '_' +os.path.basename(file)))
        else:
            img_draw.save(os.path.join(result_path, str(crop_inx) + '_no_faces_' + os.path.basename(file)))


    else:
        x_aligned, prob = mtcnn(img, return_prob=True)
        if x_aligned is not None:
            if plot_cropped_faces:
                for crop_inx in range(x_aligned.shape[0]):
                    face_tens = x_aligned[crop_inx,:,:,:].squeeze().permute(1,2,0).cpu().numpy()
                    img2 = cv2.cvtColor(face_tens, cv2.COLOR_RGB2BGR)
                    normalizedImg = np.zeros_like(img2)
                    normalizedImg = cv2.normalize(img2, normalizedImg, 0, 255, cv2.NORM_MINMAX)
                    img2 = normalizedImg.astype('uint8')
                    window_name = os.path.basename(file)
                    # cv2.imshow(window_name, img)

                    # cv2.setWindowTitle(window_name, str(movie_id) + '_mdf_' + str(mdf) + '_' + caption)
                    # cv2.putText(image, caption + '_ prob_' + str(lprob.sum().__format__('.3f')) + str(lprob),
                    #             fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 255), thickness=2,
                    #             lineType=cv2.LINE_AA, org=(10, 40))
                    cv2.imwrite(os.path.join(result_path, str(crop_inx) + '_' +os.path.basename(file)), img2)  # (image * 255).astype(np.uint8))#(inp * 255).astype(np.uint8))

        


            # plt.imshow(face_tens)
            # plt.savefig(os.path.join(result_path, os.path.basename(file)))
        print('Face detected with probability: {}'.format(prob))
        aligned.append(x_aligned)
        # names.append(dataset.idx_to_class[y])

aligned = torch.stack(aligned).to(device)
embeddings = resnet(aligned).detach().cpu()
dists = [[(e1 - e2).norm().item() for e2 in embeddings] for e1 in embeddings]
for mdfs_ix in range(len(aligned)):
    similar_face_mdf = np.argmin(np.array(dists[mdfs_ix])[np.where(np.array(dists[mdfs_ix])!=0)]) # !=0 is the (i,i) items which is one vs the same


"""
def plot_vg_over_image(result, frame_, caption, lprob):
    import numpy as np
    print("SoftMax score of the decoder", lprob, lprob.sum())
    print('Caption: {}'.format(caption))
    window_name = 'Image'
    image = np.array(frame_)
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    normalizedImg = np.zeros_like(img)
    normalizedImg = cv2.normalize(img, normalizedImg, 0, 255, cv2.NORM_MINMAX)
    img = normalizedImg.astype('uint8')

    image = cv2.rectangle(
        img,
        (int(result[0]["box"][0]), int(result[0]["box"][1])),
        (int(result[0]["box"][2]), int(result[0]["box"][3])),
        (0, 255, 0),
        3
    )
    # print(caption)
    movie_id = '111'
    mdf = '-1'
    path = './'
    file = 'pokemon'
    cv2.imshow(window_name, img)

    cv2.setWindowTitle(window_name, str(movie_id) + '_mdf_' + str(mdf) + '_' + caption)
    cv2.putText(image, caption + '_ prob_' + str(lprob.sum().__format__('.3f')) + str(lprob),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 255), thickness=2,
                lineType=cv2.LINE_AA, org=(10, 40))
    fname = str(file) + '_' + str(caption) + '.png'
    cv2.imwrite(os.path.join(path, fname),
                image)  # (image * 255).astype(np.uint8))#(inp * 255).astype(np.uint8))


"""