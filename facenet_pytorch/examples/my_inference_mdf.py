from logging import warning
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
import matplotlib.pyplot as plt
import tqdm
import pickle
from PIL import Image, ImageDraw
import PIL.ImageFont as ImageFont
import PIL.ImageColor as ImageColor
import matplotlib.colors as mcolors

import cv2
import re
from collections import Counter

# test
# from nebula3_reid.facenet_pytorch.examples.clustering import dbscan_cluster, _chinese_whispers
from examples.clustering import dbscan_cluster, _chinese_whispers
from facenet_pytorch import MTCNN, InceptionResnetV1, extract_face, fixed_image_standardization
# from facenet_pytorch.models import mtcnn, inception_resnet_v1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from argparse import ArgumentParser
from face_reid_util import p_r_plot_multi_class, umap_plot
import torchvision.transforms as T
transform = T.ToPILImage()


class EmbeddingsCollect():
    def __init__(self):
        self.embed = list()
        self.label = list()
        return

try:
    font = ImageFont.truetype('arial.ttf', 24)
except IOError:
    font = ImageFont.truetype("Tests/fonts/FreeMono.ttf", 84) #ImageFont.load_default()

color_space = [ImageColor.getrgb(n) for n, c in ImageColor.colormap.items()][1:] # avoid th aliceblue a light white one
    # [ImageColor.getrgb('blue'), ImageColor.getrgb('green'),
    #             ImageColor.getrgb('brown'), ImageColor.getrgb('red'),
    #             ImageColor.getrgb('orange'), ImageColor.getrgb('black'),
    #             ImageColor.getrgb('LightGray'),ImageColor.getrgb('white'),
    #            ImageColor.getrgb('aliceblue'), ImageColor.getrgb('antiquewhite') ]

not_id = -1
# [n for n, c in ImageColor.colormap.items()]

# color_cont = [tuple(mcolors.hsv_to_rgb((0.33+(1-x), 1, 255)).astype('int')) for x in np.arange(0, 1, 0.01)]


def calculate_ap(annotation_path, mdf_face_id_all, result_path, movie_name):
    gt_vs_det, all_no_det_clip_key = parse_annotations_lsmdc(annotation_path, mdf_face_id_all, movie_name)

    most_common = dict()
    remove_prev_ids = list()
    for key, value in gt_vs_det.items():
        array_det_indecses = gt_vs_det[key][gt_vs_det[key] != -1]
        # remove previous most common from
        for rem_id in remove_prev_ids:
            array_det_indecses = array_det_indecses[array_det_indecses != rem_id]
        # array_det_indecses = [i[1][i[1] != -1] for i in gt_vs_det.items()][_gt_id]
        if any(array_det_indecses): # may be no assignment  : Counter([i[1][i[1] != -1] for i in gt_vs_det.items()][0])
            most_common_detected_index_assigned = Counter(array_det_indecses).most_common(1)[0][0] # take most frequent
            most_common.update({key: most_common_detected_index_assigned})
            remove_prev_ids.append(most_common_detected_index_assigned)

    for key, value in gt_vs_det.items(): #
        lsmdc_person_id = most_common.get(key)
        if lsmdc_person_id is not None:
            ratio_of_the_matching_labels = np.array((value == lsmdc_person_id)).astype('int').sum()/value.shape[0]
            print("LSMDC PERSON {} ratio of matching labels {}".format(lsmdc_person_id, ratio_of_the_matching_labels))
    # instance-level accuracy over ID pairs (“Inst-Acc”) as Trevor paper identity aware...
    """
Note, that it is important to
correctly predict both “Same ID” and “Different ID” labels, which can be seen
as a 2-class prediction problem. The instance-level accuracy does not distinguish
between these two cases. Thus, we introduce a class-level accuracy, where we
separately compute accuracy over the two subsets of ID pairs (“Same-Acc”,
“Diff-Acc”) and report the harmonic mean between the two (“Class-Acc”).

    """
    # n_classes = np.unique()
    # all_targets = list()
    # all_predictions = list()
    # p_r_plot_multi_class(all_targets, all_predictions, result_path, thresholds_every_in=5, unique_id=None,
    #                      classes=[*range(n_classes)])

    df_all_no_det_clip_key = pd.DataFrame(all_no_det_clip_key)
    df_all_no_det_clip_key.to_csv(os.path.join(result_path, 'fn_mdf_list.csv'), index=False)
    return


def parse_annotations_lsmdc(annotation_path, mdf_face_id_all, movie_name):
    df = pd.read_csv(annotation_path, index_col=False)  # , dtype={'id': 'str'})
    df['movie'] = df['clip'].apply(lambda x: "_".join(x.split('-')[0].split('.')[0].split('_')[:-1]))
    df = df[df.movie == movie_name]
    print("Total No of movies", len(df['movie'].unique()))
    gt_vs_det = dict() # collect the detected IDs only when single PERSONx GT exist,

    for movie in df['movie'].unique():
        no_detected_id_mdfs = 0
        all_no_det_clip_key = list()
        for clip in df['clip'][df.movie == movie]:
            # print("Movie /clip", movie, clip)

            mdf_keys_per_lsmdc_clip = [key for key, value in mdf_face_id_all.items() if clip.lower() in key.lower()]
            if not (mdf_keys_per_lsmdc_clip):
                Warning("No MDF was found although caption based annotations exist", movie, clip)
                continue
 # 1 LSMDC clip may contains few MDF which one of them may hold ID equivalent to GT in case more than 1 the same loop iterate multiple time but the final update will take according to the right MDF key
            predicted_per_clip = list()
            for clip_key in mdf_keys_per_lsmdc_clip:
                # Parse annotation from CSV can be single or multiple persons per clip = multiple MDFs
                # df['id'][df['clip']==clip]
                ids1 = df['id'][df['clip'] == clip].item()
                ids1 = ids1.replace('[', '')
                ids1 = ids1.replace(']', '')
                start_vec = [i.span() for i in re.finditer('person', ids1.lower())]  # [(), ()]
                if not (start_vec):
                    id_no_person_per_mdf = list() # empty dummy to skip the next merge
                    continue

                commas = [i.span() for i in re.finditer(',', ids1.lower())]
                commas.append((len(ids1), len(ids1)))  # dummy for last  seperator at the end of 2nd person

                id_no_person_per_mdf = list() # LSMDC one annotation of PERSONS per all CLIP no idea about MDFs
                for strt_stop, comma in zip(start_vec, commas):
                    # start = ids1.lower().find('person')
                    start = strt_stop[1]

                    # stop = strt_stop[0]
                    if start != -1:
                        id_no_person_per_mdf.append(int(ids1[start:comma[0]]))
                    else:
                        continue

                predicted_per_clip.extend([v['id'] for k, v in mdf_face_id_all[clip_key].items() if 'id' in v])
                mdf_face_id_all[clip_key].update({'gt_from_caption': id_no_person_per_mdf})
                mdf_face_id_all[clip_key].update({'clip_name': clip}) # add the common clip name shared among few MDF records
                # print(ids1, id_no_person_per_mdf)
# Merge : per lsmdc clip review predicted IDs to be uniqye in order not to solve bipartite matching
            if (not(predicted_per_clip) or all([x==-1 for x in predicted_per_clip])) and id_no_person_per_mdf:
                predicted_per_clip = np.array([not_id]) # class dummy means no ID at all
                print('FN !! CLIP: {} MDF : {} '.format(clip, clip_key))
                all_no_det_clip_key.append(clip_key)
                no_detected_id_mdfs += 1
            else:
                predicted_per_clip = np.unique(predicted_per_clip)
            if (predicted_per_clip.shape[0]> 1 or len(id_no_person_per_mdf)>1) or not(id_no_person_per_mdf):
                # print('No unique predicted IDs or GT PERSONS can not tell skip ')
                continue
            id_no_person_per_mdf = id_no_person_per_mdf[0]
            # Update dict mapping of PERSON GT to related detected IDs over all LSMDC clips
            if id_no_person_per_mdf in gt_vs_det: # for the first time indexing that gt
                predicted_per_clip = np.append(predicted_per_clip, gt_vs_det[id_no_person_per_mdf])
            gt_vs_det.update({id_no_person_per_mdf: predicted_per_clip})

        print("Missing IDs {} out of {} MDFs ratio: {}".format(no_detected_id_mdfs, len(mdf_face_id_all), no_detected_id_mdfs/len(mdf_face_id_all)))
        return gt_vs_det, all_no_det_clip_key


def plot_id_over_mdf(mdf_id_all, result_path, path_mdf, plot_fn=False): # FN plot the IDs that weren't classified
    text_width, text_height = font.getsize('ID - 1')

    for file, ids_desc_all_clip_mdfs in tqdm.tqdm(mdf_id_all.items()):
        file_path = os.path.join(path_mdf, file)
        img = Image.open(file_path)
        img_draw = img.copy()
        draw = ImageDraw.Draw(img_draw)
        for ids, bbox_n_id in ids_desc_all_clip_mdfs.items():
            if ids_desc_all_clip_mdfs[ids]['id'] != -1:
                box = ids_desc_all_clip_mdfs[ids]['bbox']
                draw.rectangle(box.tolist(), width=10, outline=color_space[ids_desc_all_clip_mdfs[ids]['id'] % len(color_space)]) # landmark plot
                margin = np.ceil(0.05 * text_height)
                draw.text(
                    (box[0] + margin, box[3] - text_height - margin),
                    str(ids_desc_all_clip_mdfs[ids]['id']),
                    fill='yellow',
                    font=font)
            elif plot_fn:
                box = ids_desc_all_clip_mdfs[ids]['bbox']
                # draw.rectangle(box.tolist(), width=10, outline=color_space[ids_desc_all_clip_mdfs[ids]['id'] % len(color_space)]) # landmark plot
                draw.rounded_rectangle(box.tolist(), width=10, radius=10, outline=color_space[ids_desc_all_clip_mdfs[ids]['id'] % len(color_space)]) # landmark plot
                margin = np.ceil(0.05 * text_height)
                draw.text(
                    (box[0] + margin, box[3] - text_height - margin),
                    str(-1),
                    fill='yellow',
                    font=font)

        img_draw.save(os.path.join(result_path, 're-id_' + os.path.basename(file)))
    return

def facenet_embeddings(aligned, batch_size, image_size, device, neural_net):
# FaceNet create embeddings
    if isinstance(aligned, list):
        aligned = torch.stack(aligned)
    elif isinstance(aligned, torch.Tensor):
        pass
    else:
        raise
    if aligned.shape[0]%batch_size != 0: # all images size are Int multiple of batch size 
        pad = batch_size - aligned.shape[0]%batch_size
    else:
        pad = 0
    aligned = torch.cat((aligned, torch.zeros((pad, 3, image_size, image_size))), 0)
    all_embeddings = list()
    for frame_num in range(int(aligned.shape[0]/batch_size)):
        with torch.no_grad():
            if batch_size > 0:
                # ind = frame_num % batch_size
                batch_array = aligned[frame_num*batch_size:(frame_num+1)*batch_size, :,:,:]
                batch_array = batch_array.to(device)
                embeddings = neural_net(batch_array).detach().cpu()
                all_embeddings.append(embeddings)
            else:
                embeddings = neural_net(batch_array)
                all_embeddings.append(embeddings)
    all_embeddings = torch.cat(all_embeddings, 0) #np.concatenate(all_embeddings)
    all_embeddings = all_embeddings[:all_embeddings.shape[0]-pad,:]

    return all_embeddings

def find_key_given_value(clusters, ix):
    # [list(clusters.values())[0].tolist().index(ix)]
    for id, bbox_no in clusters.items():  # for name, age in dictionary.iteritems():  (for Python 2.x)
        if ix in bbox_no:
            return id 
    return -1


def extract_faces(path_mdf, result_path, result_path_good_resolution_faces, margin=0, 
                    batch_size=128, min_face_res=64,
                    prob_th_filter_blurr=0.95, re_id_method={'method':'dbscan', 'cluster_threshold': 0.3},
                    plot_cropped_faces=False):
    # rel_path = 'nebula3_reid/facenet_pytorch'

    # TODO result_path_good_resolution_faces_frontal =  # filter profile
    # plt.savefig('/home/hanoch/notebooks/nebula3_reid/face_tens.png')
    if result_path and not os.path.exists(result_path):
        os.makedirs(result_path)

    if result_path_good_resolution_faces and not os.path.exists(result_path_good_resolution_faces):
        os.makedirs(result_path_good_resolution_faces)

    workers = 0 if os.name == 'nt' else 4
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))
    #### Define MTCNN module
    """
    Default params shown for illustration, but not needed. Note that, since MTCNN is a collection of neural nets and other code, the device must be passed in the following way to enable copying of objects when needed internally.

    See `help(MTCNN)` for more details.

    """

    detection_with_landmark = False
    if plot_cropped_faces:
        print("FaceNet output is post process : fixed_image_standardization")

    keep_all = True # HK revert
    
    save_images = True
    # post_process=True => fixed_image_standardization 
    mtcnn = MTCNN(
        image_size=160, margin=margin, min_face_size=min_face_res,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True, keep_all=keep_all, 
        device=device ) #post_process=False
    # Modify model to VGGFace based and resnet
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    aligned = list()
    names = list()
    mtcnn_cropped_image = list()

    filenames = [os.path.join(path_mdf, x) for x in os.listdir(path_mdf)
                        if x.endswith('png') or x.endswith('jpg')]
    if not bool(filenames):
        raise ValueError('No files at that folder')
    
    mdf_id_all = dict()
    for file_inx, file in enumerate(tqdm.tqdm(filenames)):
        img = Image.open(file) # '3001_21_JUMP_STREET_00.03.13.271-00.03.16.551'

        if 0: # direct
            x_aligned, prob = mtcnn(img, return_prob=True)
        else:
            try:# Face landmarks + embeddings of aligned face
                batch_boxes, prob, lanmarks_points = mtcnn.detect(img, landmarks=True)
                x_aligned = mtcnn.extract(img, batch_boxes, save_path=None) # implicitly saves the faces
            except Exception as ex:
                print(ex)
                continue

        face_id = dict()
        if x_aligned is not None:
            if len(x_aligned.shape) == 3:
                x_aligned = x_aligned.unsqueeze(0)
                prob = np.array([prob])
            for crop_inx in range(x_aligned.shape[0]):
                if prob[crop_inx] > prob_th_filter_blurr:
                    face_tens = x_aligned[crop_inx, :, :, :].squeeze().permute(1, 2, 0).cpu().numpy()
                    face_bb_resolution = 'res_ok'
                    # for p in lanmarks_points:
                    #     draw.rectangle((p - 1).tolist() + (p + 1).tolist(), width=2)

                    img2 = cv2.cvtColor(face_tens, cv2.COLOR_RGB2BGR)  #???? RGB2BGR
                    img2 = cv2.cvtColor(face_tens, cv2.COLOR_BGR2RGB) # undo 
                    # img2 = Image.fromarray((face_tens * 255).astype(np.uint8)) 
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
                                                str(file_inx) + '_prob_' + str(prob[crop_inx].__format__('.2f')) + '_' + str(face_tens.shape[0]) + '_face_{}'.format(crop_inx) + os.path.basename(file))
                    if plot_cropped_faces: # The normalization handles the fixed_image_standardization() built in in MTCNN forward engine
                        cv2.imwrite(save_path, img2)  # (image * 255).astype(np.uint8))#(inp * 255).astype(np.uint8))

                    mtcnn_cropped_image.append(img2)
                    if 0:  # save MDFs
                        cv2.imwrite(os.path.join(result_path, str(crop_inx) + '_' +os.path.basename(file)), img2)  # (image * 255).astype(np.uint8))#(inp * 255).astype(np.uint8))
                    aligned.append(x_aligned[crop_inx,:,:,:])
                    fname = str(file_inx) + '_' + '_face_{}'.format(crop_inx) + os.path.basename(file)
                    names.append(fname)
                    face_id.update({fname: {'bbox': batch_boxes[crop_inx], 'id': -1, 'gt': -1}})
                    # print('Face detected with probability: {:8f}'.format(prob[crop_inx]))
            if bool(face_id): # Cases where none of the prob>th
                mdf_id_all.update({os.path.basename(file):face_id})
    
    all_embeddings = facenet_embeddings(aligned, batch_size, 
                                                    image_size=mtcnn.image_size, device=device, neural_net=resnet)

    return all_embeddings, mtcnn_cropped_image, names, mdf_id_all
    # embeddings = resnet(aligned).detach().cpu()
    ##TODO: try cosine similarity 
    ## TODO : vs GT add threshold -> calc Precision recall infer threshold->run over testset
def re_identification(all_embeddings, mtcnn_cropped_image, names,
                    re_id_method, mdf_id_all, result_path):

    dbscan_result_path = os.path.join(result_path, 'dbscan')
    if dbscan_result_path and not os.path.exists(dbscan_result_path):
        os.makedirs(dbscan_result_path)

    if re_id_method['method'] == 'similarity':
        top_k = 3
        dists = [[(e1 - e2).norm().item() for e2 in all_embeddings] for e1 in all_embeddings]
        # all_similar_face_mdf = list()
        dist_per_face = torch.from_numpy(np.array(dists).astype('float32'))
        v_top_k, i_topk = torch.topk(-dist_per_face, k=top_k, dim=1) # topk of -dist is mink of dist idenx 0 is 1 vs. the same 1.
        for i in range(all_embeddings.shape[0]):
            for t in range(top_k):
                print("pairwise match to face {} : {} is {} \n ".format(i, names[i], names[i_topk[i][t]]))
        # for mdfs_ix in range(all_embeddings.shape[0]):
        #     similar_face_mdf = np.argmin(np.array(dists[mdfs_ix])[np.where(np.array(dists[mdfs_ix])!=0)]) # !=0 is the (i,i) items which is one vs the same
        #     all_similar_face_mdf.append(similar_face_mdf)
    elif re_id_method['method'] == 'dbscan':
        clusters = dbscan_cluster(labels=names, images=mtcnn_cropped_image, matrix=all_embeddings,
                        out_dir=dbscan_result_path, cluster_threshold=re_id_method['cluster_threshold'],
                        min_cluster_size=re_id_method['min_cluster_size'], metric='cosine')
        # when cosine dist ->higher =>more out of cluster(non core points) are gathered and became core points as in clusters hence need to increase the K-NN, cluster size
        n_clusters = len([i[0] for i in clusters.items()])

        if n_clusters <= 2:
            warning("too few classes ")

        print("Total {} clusters and total IDs {}".format(n_clusters,
                                        np.concatenate([x[1] for x in clusters.items()]).shape[0]))
    else:
        raise
    labeled_embed = EmbeddingsCollect()

    for mdf, id_v in mdf_id_all.items():
        for k, v in id_v.items():
            if k in names:
                ix = names.index(k)
                id_cluster_no = find_key_given_value(clusters, ix)
                if id_cluster_no != -1:
                    mdf_id_all[mdf][k]['id'] = id_cluster_no
                    labeled_embed.embed.append(all_embeddings[ix])
                    labeled_embed.label.append(id_cluster_no)

    return mdf_id_all, labeled_embed


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

# '3001_21_JUMP_STREET'--task metric_calc --cluster-threshold 0.3 --min-face-res 72 --min-cluster-size 6
    parser = ArgumentParser()
    parser.add_argument("--path-mdf", type=str, help="MVAD dataset path",  default='/home/hanoch/mdf_lsmdc/all')
    parser.add_argument("--movie", type=str, help="MVAD-Names dataset file path", default='0001_American_Beauty')#'3001_21_JUMP_STREET')#default='0001_American_Beauty')
    # parser.add_argument("--movie", type=str, help="Name of the movie to process.", default=None)
    # parser.add_argument("--clip", type=str, help="Clip IDs (split by space).", default=None)
    parser.add_argument("--result-path", type=str, help="", default='/home/hanoch/results/face_reid/face_net')
    parser.add_argument('--batch-size', type=int, default=128, metavar='INT', help="TODO")
    parser.add_argument('--mtcnn-margin', type=int, default=40, metavar='INT', help="TODO")
    parser.add_argument('--min-face-res', type=int, default=64, metavar='INT', help="TODO")
    parser.add_argument('--cluster-threshold', type=float, default=0.28, metavar='FLOAT', help="TODO")
    parser.add_argument('--min-cluster-size', type=int, default=5, metavar='INT', help="TODO")
    parser.add_argument('--task', type=str, default='classify_faces', choices=['classify_faces', 'metric_calc', 'embeddings_viz_umap', 'plot_id_over_mdf'], metavar='STRING',
                        help='')
    
    parser.add_argument("--annotation-path", type=str, help="",  default='/home/hanoch/notebooks/nebula3_reid/annotations/LSMDC16_annos_training_onlyIDs_NEW_local.csv')
    
    args = parser.parse_args()

    result_path = os.path.join(args.result_path, args.movie)
    path_mdf = args.path_mdf
    path_mdf = os.path.join(path_mdf, args.movie)

    result_path_good_resolution_faces = os.path.join(result_path, 'good_res')
    batch_size = args.batch_size
    margin = args.mtcnn_margin
    min_face_res = args.min_face_res#64 #+ margin #64*1.125 + margin # margin is post processing 
    prob_th_filter_blurr = 0.95
    batch_size = batch_size if torch.cuda.is_available() else 0
    re_id_method = {'method': 'dbscan', 'cluster_threshold': args.cluster_threshold, 'min_cluster_size': args.min_cluster_size}#0.4}

    result_path = os.path.join(result_path, 'res_' + str(min_face_res) + '_margin_' + str(margin) + '_eps_'  + str(args.cluster_threshold)) + '_KNN_'+ str(re_id_method['min_cluster_size'])
    if args.task == 'classify_faces':
        all_embeddings, mtcnn_cropped_image, names, mdf_id_all = extract_faces(path_mdf, result_path, result_path_good_resolution_faces,
                                                                margin=margin, batch_size=batch_size, min_face_res=min_face_res,
                                                                prob_th_filter_blurr=prob_th_filter_blurr, re_id_method=re_id_method)

        mdf_id_all, labeled_embed = re_identification(all_embeddings, mtcnn_cropped_image, names, re_id_method, mdf_id_all, result_path)

        if re_id_method['method'] == 'dbscan':
            with open(os.path.join(result_path, 're-id_res_' + str(min_face_res) + '_' + str(prob_th_filter_blurr) + '_eps_' + str(re_id_method['cluster_threshold']) + '_KNN_'+ str(re_id_method['min_cluster_size']) +'.pkl'), 'wb') as f:
                pickle.dump(mdf_id_all, f)
            if 1:
                with open(os.path.join(result_path, 'face-id_embeddings_embed_' + str(min_face_res) + '_' + str(prob_th_filter_blurr) + '_eps_' + str(re_id_method['cluster_threshold']) + '_KNN_'+ str(re_id_method['min_cluster_size']) + '.pkl'), 'wb') as f1:
                    pickle.dump(labeled_embed.embed, f1)

            with open(os.path.join(result_path, 'face-id_embeddings_label_' + str(min_face_res) + '_' + str(prob_th_filter_blurr) + '_eps_' + str(re_id_method['cluster_threshold']) + '_KNN_'+ str(re_id_method['min_cluster_size']) + '.pkl'), 'wb') as f1:
                pickle.dump(labeled_embed.label, f1)

        else:
            raise

        re_id_result_path = os.path.join(result_path, 're_id')
        if re_id_result_path and not os.path.exists(re_id_result_path):
            os.makedirs(re_id_result_path)

        plot_id_over_mdf(mdf_id_all, result_path=re_id_result_path, path_mdf=path_mdf)


    
    elif args.task == 'metric_calc':    
        import pandas as pd
        with open(os.path.join(result_path, 're-id_res_' + str(min_face_res) + '_' + str(prob_th_filter_blurr) + '_eps_' + str(args.cluster_threshold) + '_KNN_'+ str(args.min_cluster_size) + '.pkl'), 'rb') as f:
            mdf_face_id_all = pickle.load(f)


        ap = calculate_ap(args.annotation_path, mdf_face_id_all, result_path, args.movie)

    elif args.task == 'embeddings_viz_umap':
        labeled_embed = EmbeddingsCollect()

        with open(os.path.join(result_path, 'face-id_embeddings_embed_' + str(min_face_res) + '_' + str(prob_th_filter_blurr) + '_eps_' + str(args.cluster_threshold) + '_KNN_'+ str(args.min_cluster_size) + '.pkl'), 'rb') as f:
            labeled_embed.embed = pickle.load(f)

        with open(os.path.join(result_path, 'face-id_embeddings_label_' + str(min_face_res) + '_' + str(prob_th_filter_blurr) + '_eps_' + str(re_id_method['cluster_threshold']) + '_KNN_'+ str(re_id_method['min_cluster_size']) + '.pkl'), 'rb') as f1:
            labeled_embed.label = pickle.load(f1)

        plot_fn = True
        if plot_fn:
            import pandas as pd
            path_fn = '/home/hanoch/notebooks/nebula3_reid/annotations/FN_0001_American_Beauty.csv'
            df = pd.read_csv(path_fn, index_col=False)  # , dtype={'id': 'str'})
            df.dropna(axis='columns')
            mdf_path = '/home/hanoch/results/face_reid/face_net/0001_American_Beauty/fn'
            id_fn = 1
            print("FN IDs is ", df[df['id'] == id_fn]['id_name'].unique())
            if 0:
                import subprocess
                if path_mdf is None:
                    raise
                for reid_fname in df[df['id'] == 1]['mdf']:  # Kevin Spacy
                    fname = reid_fname.split('re-id_')[-1]
                    file_full_path = subprocess.getoutput('find ' + path_mdf + ' -iname ' + '"*' + fname + '*"')
                    if not file_full_path:
                        print("File not found", fname)
                    dest_path = '/home/hanoch/results/face_reid/face_net/0001_American_Beauty/fn'
                    subprocess.getoutput('cp -p ' + file_full_path + ' ' + dest_path)

            result_path_fn = os.path.join(mdf_path, 'face_rec')
            if result_path_fn and not os.path.exists(result_path_fn):
                os.makedirs(result_path_fn)

            all_embeddings, mtcnn_cropped_image, names, mdf_id_all = extract_faces(path_mdf=mdf_path, result_path=result_path_fn,
                                                                                    result_path_good_resolution_faces=result_path_fn,
                                                                                    margin=margin, batch_size=batch_size,
                                                                                    min_face_res=min_face_res,
                                                                                    prob_th_filter_blurr=prob_th_filter_blurr,
                                                                                    re_id_method=re_id_method['method'],
                                                                                    plot_cropped_faces=True)

            print("Embeddings of all reID ", len(labeled_embed.embed))
            # Collect statistics of each ID
            test_embed = all_embeddings.cpu().numpy()
            for id_ix in np.sort(np.unique(labeled_embed.label)):
                id_embed = [labeled_embed.embed[i] for i in id_ix]
                id_mean = np.mean(np.stack(id_embed), axis=0)

                dist_euc = np.linalg.norm(test_embed - id_mean)
                cos_dist = 1 - np.sum(test_embed*id_mean)/(np.linalg.norm(test_embed) * np.linalg.norm(id_mean))
            # Add the unclassified ID
            fn_unique_label = np.sort(np.unique(labeled_embed.label))[-1] + 1
            for ix in range(all_embeddings.shape[0]):
                labeled_embed.embed.append(all_embeddings[ix])
                labeled_embed.label.append(fn_unique_label)


            print("total embeddings of all reID and FN ",len(labeled_embed.embed))
            # FN class Already known no need for re-id
            result_path = result_path_fn

            id_ix = np.where(np.array(labeled_embed.label) == id_fn)[0]
        umap_plot(labeled_embed, result_path)

    elif args.task == 'plot_id_over_mdf':
        with open(os.path.join(result_path,
                               're-id_res_' + str(min_face_res) + '_' + str(prob_th_filter_blurr) + '_eps_' + str(
                                       args.cluster_threshold) + '_KNN_' + str(args.min_cluster_size) + '.pkl'), 'rb') as f:
            mdf_face_id_all = pickle.load(f)

        re_id_result_path = os.path.join(result_path, 're_id')
        if 1:
            re_id_result_path = os.path.join(result_path, 're_id_fn')

            if re_id_result_path and not os.path.exists(re_id_result_path):
                os.makedirs(re_id_result_path)

            plot_fn = True
            plot_id_over_mdf(mdf_face_id_all, result_path=re_id_result_path, path_mdf=path_mdf, plot_fn=plot_fn)
        else:
            plot_id_over_mdf(mdf_face_id_all, result_path=re_id_result_path, path_mdf=path_mdf)


    return

    # with open(os.path.join(result_path, 're-id_res_' + str(min_face_res) + '_' + str(prob_th_filter_blurr) + '.pkl'), 'rb') as f:
    #     mdf_id_all = pickle.load(f)
    all_det_similarity = classify_min_dist(faces_path=result_path_good_resolution_faces, batch_size=batch_size)
    all_det_cluster = classify_clustering(faces_path=result_path_good_resolution_faces, batch_size=batch_size)

if __name__ == '__main__':
    main()


"""
--task metric_calc --cluster-threshold 0.28 --min-face-res 64 --min-cluster-size 5 --movie 0001_American_Beauty
--task metric_calc --cluster-threshold 0.3 --min-face-res 64 --min-cluster-size 5 --movie '3001_21_JUMP_STREET'
--task classify_faces --cluster-threshold 0.28 --min-face-res 64 --min-cluster-size 5

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

"""
