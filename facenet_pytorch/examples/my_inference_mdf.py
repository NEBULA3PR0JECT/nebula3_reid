import sys
import os
import copy

sys.path.append('/home/hanoch/notebooks/nebula3_reid')
sys.path.append('/home/hanoch/notebooks/nebula3_reid/facenet_pytorch')
sys.path.append('/home/hanoch/notebooks/nebula3_reid/facenet_pytorch/examples')
curr_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(curr_dir)
directory = os.path.abspath(__file__)
# # setting path
# sys.path.append(directory.parent.parent)

os.path.abspath(os.path.join(__file__, os.pardir))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
from PIL import Image, ImageDraw
import cv2
# test
# from nebula3_reid.facenet_pytorch.examples.clustering import dbscan_cluster, _chinese_whispers
from examples.clustering import dbscan_cluster, _chinese_whispers
from facenet_pytorch import MTCNN, InceptionResnetV1, extract_face, fixed_image_standardization
# from facenet_pytorch.models import mtcnn, inception_resnet_v1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as T
transform = T.ToPILImage()


def extract_faces(path_mdf, result_path, result_path_good_resolution_faces, margin=0, 
                    batch_size=128, min_face_res = 64,
                    prob_th_filter_blurr=0.95):
    # rel_path = 'nebula3_reid/facenet_pytorch'

    # TODO result_path_good_resolution_faces_frontal =  # filter profile
    # plt.savefig('/home/hanoch/notebooks/nebula3_reid/face_tens.png')
    if result_path and not os.path.exists(result_path):
        os.makedirs(result_path)

    if result_path_good_resolution_faces and not os.path.exists(result_path_good_resolution_faces):
        os.makedirs(result_path_good_resolution_faces)

    dbscan_result_path = os.path.join(result_path, 'dbscan')
    if dbscan_result_path and not os.path.exists(dbscan_result_path):
        os.makedirs(dbscan_result_path)

    workers = 0 if os.name == 'nt' else 4
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))
    #### Define MTCNN module
    """
    Default params shown for illustration, but not needed. Note that, since MTCNN is a collection of neural nets and other code, the device must be passed in the following way to enable copying of objects when needed internally.

    See `help(MTCNN)` for more details.

    """
    plot_cropped_faces = True
    detection_with_landmark = False
    if plot_cropped_faces:
        print("FaceNet output is post process : fixed_image_standardization")

    keep_all = True # HK revert
    
    save_images = True
    # post_process=True => fixed_image_standardization 
    mtcnn = MTCNN(
        image_size=160, margin=margin, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True, keep_all=keep_all, 
        device=device ) #post_process=False
    # Modify model to VGGFace based and resnet
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    # def collate_fn(x):
    #     return x[0]

    # dataset = datasets.ImageFolder(os.path.join(rel_path, 'data/test_images'))
    # dataset = datasets.ImageFolder(path_mdf)
    # dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
    # loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)


    if 0:
        def collate_fn(x):
            return x[0]
        workers = 0 if os.name == 'nt' else 4
        dataset = datasets.ImageFolder(path_mdf)
        dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
        loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)

    aligned = list()
    names = list()
    mtcnn_cropped_image = list()

    filenames = [os.path.join(path_mdf, x) for x in os.listdir(path_mdf)
                        if x.endswith('png') or x.endswith('jpg')]
    if filenames is None:
        raise ValueError('No files at that folder')

    for file_inx, file in enumerate(tqdm.tqdm(filenames)):
        img = Image.open(file)#('images/office1.jpg')
    # for file_inx, (x, y) in enumerate(loader):
    #     img = x
    #     file = dataset.imgs[file_inx][0]

        # if 1:
        #     img = Image.fromarray(np.array(img)[:, :, ::-1])
        # if file_inx == 50:
        #     break
        if detection_with_landmark:
            boxes, probs, points = mtcnn.detect(img, landmarks=True)
            if boxes is not None:
            # Draw boxes and save faces
                img_draw = img.copy()
                draw = ImageDraw.Draw(img_draw)
                for i, (box, point) in enumerate(zip(boxes, points)):

                    print("confidence: {}".format(str(probs[0].__format__('.2f'))))
                    if (box[2] - box[0])>min_face_res and (box[3] - box[1])>min_face_res:
                        face_bb_resolution = 'res_ok'
                    else:
                        face_bb_resolution = 'res_bad'

                    draw.rectangle(box.tolist(), width=5) # landmark plot
                    for p in point:
                        # draw.rectangle((p - 10).tolist() + (p + 10).tolist(), width=4)
                        draw.rectangle((p - 1).tolist() + (p + 1).tolist(), width=2)

                        # for i in range(5):
                        #     draw.ellipse([
                        #         (p[i] - 1.0, p[i + 5] - 1.0),
                        #         (p[i] + 1.0, p[i + 5] + 1.0)
                        #     ], outline='blue')


                    print("N-Landmarks {}".format(point.shape[0]))

                    if save_images:
                        if face_bb_resolution == 'res_ok':
                            save_path = os.path.join(result_path_good_resolution_faces, 
                                                        str(file_inx) + '_' + face_bb_resolution + '_' + str(point.shape[0]) + '_face_{}'.format(i) + os.path.basename(file))
                        else:
                            save_path = os.path.join(result_path, 
                                            str(file_inx) + '_' + face_bb_resolution + '_' + str(point.shape[0]) + '_face_{}'.format(i) + os.path.basename(file))
                    else:
                        save_path = None
# From some reason the channels are BGR save_img() revert that back
                    x_aligned = extract_face(img, box, save_path=save_path) # implicitly saves the faces
                    # imgp = transform(x_aligned.squeeze())
                    # imgp.save(save_path)

                    if face_bb_resolution == 'res_ok':
                        mtcnn_cropped_image.append(x_aligned) 
                        x_aligned = fixed_image_standardization(x_aligned) # as done by the mtcnn.forward() to inject to FaceNet
                        aligned.append(x_aligned)
                        names.append(str(file_inx) + '_' + face_bb_resolution + '_'+ 'face_{}'.format(i) + os.path.basename(file))
                if save_images:
                    img_draw.save(os.path.join(result_path, str(file_inx) + '_' +os.path.basename(file)))
            else:
                img_draw.save(os.path.join(result_path, str(file_inx) + '_no_faces_' + os.path.basename(file)))


        else:
            x_aligned, prob = mtcnn(img, return_prob=True)
            if x_aligned is not None:
                if len(x_aligned.shape) ==3 :
                    x_aligned = x_aligned.unsqueeze(0)
                    prob = np.array([prob])
                for crop_inx in range(x_aligned.shape[0]):
                    if prob[crop_inx]>prob_th_filter_blurr:
                        face_tens = x_aligned[crop_inx,:,:,:].squeeze().permute(1,2,0).cpu().numpy()
                        face_bb_resolution = 'res_ok'
                        if plot_cropped_faces: # The normalization handles the fixed_image_standardization() built in in MTCNN forward engine
                            img2 = cv2.cvtColor(face_tens, cv2.COLOR_RGB2BGR)  #???? RGB2BGR
                            normalizedImg = np.zeros_like(img2)
                            normalizedImg = cv2.normalize(img2, normalizedImg, 0, 255, cv2.NORM_MINMAX)
                            img2 = normalizedImg.astype('uint8')
                            window_name = os.path.basename(file)
                        # cv2.imshow(window_name, img)

                        # cv2.setWindowTitle(window_name, str(movie_id) + '_mdf_' + str(mdf) + '_' + caption)
                        # cv2.putText(image, caption + '_ prob_' + str(lprob.sum().__format__('.3f')) + str(lprob),
                        #             fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 255), thickness=2,
                        #             lineType=cv2.LINE_AA, org=(10, 40))
                            save_path = os.path.join(result_path_good_resolution_faces, 
                                                        str(file_inx) + '_prob_' + str(prob[crop_inx]) + '_' + str(face_tens.shape[0]) + '_face_{}'.format(crop_inx) + os.path.basename(file))

                            cv2.imwrite(save_path, img2)  # (image * 255).astype(np.uint8))#(inp * 255).astype(np.uint8))
                            mtcnn_cropped_image.append(img2)
                        # cv2.imwrite(os.path.join(result_path, str(crop_inx) + '_' +os.path.basename(file)), img2)  # (image * 255).astype(np.uint8))#(inp * 255).astype(np.uint8))

                        aligned.append(x_aligned[crop_inx,:,:,:])
                        names.append(str(file_inx) + '_' + face_bb_resolution + '_'+ 'face_{}'.format(crop_inx) + os.path.basename(file))


                # plt.imshow(face_tens)
                # plt.savefig(os.path.join(result_path, os.path.basename(file)))
                        print('Face detected with probability: {:8f}'.format(prob[crop_inx]))
                        # if prob[crop_inx] < 0.95:
                        #     print('ka')
            # aligned.append(x_aligned)
            # names.append(dataset.idx_to_class[y])
    if 1:
        # aligned = torch.stack(aligned).to(device)
        aligned = torch.stack(aligned)
        if aligned.shape[0]%batch_size != 0: # all images size are Int multiple of batch size 
            pad = batch_size - aligned.shape[0]%batch_size
        else:
            pad = 0
        aligned = torch.cat((aligned, torch.zeros((pad, 3, mtcnn.image_size, mtcnn.image_size))), 0)
        all_embeddings = list()
        for frame_num in range(int(aligned.shape[0]/batch_size)):
            with torch.no_grad():
                if batch_size > 0:
                    # ind = frame_num % batch_size
                    batch_array = aligned[frame_num*batch_size:(frame_num+1)*batch_size, :,:,:]
                    batch_array = batch_array.to(device)
                    embeddings = resnet(batch_array).detach().cpu()
                    all_embeddings.append(embeddings)
                else:
                    embeddings = resnet(batch_array)
                    all_embeddings.append(embeddings)
        all_embeddings = torch.cat(all_embeddings, 0) #np.concatenate(all_embeddings)
        all_embeddings = all_embeddings[:all_embeddings.shape[0]-pad,:]

        # embeddings = resnet(aligned).detach().cpu()
        ##TODO: try cosine similarity 
        ## TODO : vs GT add threshold -> calc Precision recall infer threshold->run over testset
        top_k = 3
        dists = [[(e1 - e2).norm().item() for e2 in all_embeddings] for e1 in all_embeddings]
        # all_similar_face_mdf = list()
        dist_per_face = torch.from_numpy(np.array(dists).astype('float32'))
        v_top_k, i_topk = torch.topk(-dist_per_face , k=top_k, dim=1) # topk of -dist is mink of dist idenx 0 is 1 vs. the same 1.
        for i in range(all_embeddings.shape[0]):
            for t in range(top_k):
                print("pairwise match to face {} : {} is {} \n ".format(i, names[i], names[i_topk[i][t]]))
        # for mdfs_ix in range(all_embeddings.shape[0]):
        #     similar_face_mdf = np.argmin(np.array(dists[mdfs_ix])[np.where(np.array(dists[mdfs_ix])!=0)]) # !=0 is the (i,i) items which is one vs the same
        #     all_similar_face_mdf.append(similar_face_mdf)
            
        dbscan_cluster(labels=names, images=mtcnn_cropped_image, matrix=all_embeddings, 
                        out_dir=dbscan_result_path, cluster_threshold=0.5, min_cluster_size=5,
                        metric='cosine')
        if 0:
            sorted_clusters = _chinese_whispers(all_embeddings)
        # do UMAP/TSNe
    
def classify_min_dist(faces_path, batch_size=128):
    x_aligned = fixed_image_standardization(x_aligned)
    return

def classify_clustering(faces_path, batch_size=128):
    x_aligned = fixed_image_standardization(x_aligned)
    return

def main():
    film = '0001_American_Beauty'
    result_path = os.path.join('/home/hanoch/results/face_reid/face_net', film)
    # path_mdf = '/home/hanoch/mdf_lsmdc/all/0011_Gandhi'#'/home/hanoch/mdfs2_lsmdc'
    if 1:
        path_mdf = os.path.join('/home/hanoch/mdfs2_lsmdc')
        # path_mdf = '/home/hanoch/notebooks/nebula3_reid/facenet_pytorch/data/test_images'
        # path_mdf = os.path.join('/home/hanoch/temp')
    else:
        path_mdf = os.path.join('/home/hanoch/mdf_lsmdc/all', film)
    result_path_good_resolution_faces = os.path.join(result_path, 'good_res')
    batch_size = 128
    margin = 0 #40
    min_face_res = 96 #+ margin #64*1.125 + margin # margin is post processing 
    batch_size = batch_size if torch.cuda.is_available() else 0

    features = extract_faces(path_mdf, result_path, result_path_good_resolution_faces, 
                            margin=margin, batch_size=batch_size, min_face_res=min_face_res)
    return
    all_det_similarity = classify_min_dist(faces_path=result_path_good_resolution_faces, batch_size=batch_size)
    all_det_cluster = classify_clustering(faces_path=result_path_good_resolution_faces, batch_size=batch_size)

if __name__ == '__main__':
    main()


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
