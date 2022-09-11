""" Face Cluster 
https://github.com/davidsandberg/facenet/blob/096ed770f163957c1e56efa7feeb194773920f6e/contributed/clustering.py
""" 
# import tensorflow as tf
import numpy as np
import importlib
import argparse
# import facenet
import os
import math

from scipy import misc
from sklearn.cluster import DBSCAN
import torchvision.transforms as T
transform = T.ToPILImage()


def dbscan_cluster(labels, images, matrix, out_dir, cluster_threshold=1, 
                    min_cluster_size=1, largest_cluster_only=False, metric='euclidean'):

 # DBSCAN is the only algorithm that doesn't require the number of clusters to be defined.
    db = DBSCAN(eps=cluster_threshold, min_samples=min_cluster_size, metric=metric)#, metric='precomputed')
    db.fit(matrix) #(n_samples, n_features),
    labels = db.labels_

    # get number of clusters
    no_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    print('No of clusters:', no_clusters)
    if no_clusters==0:
        print("Hueston")
    if no_clusters > 0:
        if largest_cluster_only:
            largest_cluster = 0
            for i in range(no_clusters):
                print('Cluster {}: {}'.format(i, np.nonzero(labels == i)[0]))
                if len(np.nonzero(labels == i)[0]) > len(np.nonzero(labels == largest_cluster)[0]):
                    largest_cluster = i
            print('Saving largest cluster (Cluster: {})'.format(largest_cluster))
            cnt = 1
            for i in np.nonzero(labels == largest_cluster)[0]:
                misc.imsave(os.path.join(out_dir, str(cnt) + '.png'), images[i])
                cnt += 1
        else:
            print('Saving all clusters')
            for i in range(no_clusters):
                cnt = 1
                print('Cluster {}: {}'.format(i, np.nonzero(labels == i)[0]))
                path = os.path.join(out_dir, str(i))
                if not os.path.exists(path):
                    os.makedirs(path)
                    for j in np.nonzero(labels == i)[0]:
                        imgp = transform(images[j])
                        imgp.save(os.path.join(path, str(cnt) + '.png'))

                        # misc.imsave(os.path.join(path, str(cnt) + '.png'), images[j])
                        cnt += 1
                else:
                    for j in np.nonzero(labels == i)[0]:
                        imgp = transform(images[j])
                        imgp.save(os.path.join(path, str(cnt) + '.png'))
                        # misc.imsave(os.path.join(path, str(cnt) + '.png'), images[j])
                        cnt += 1


def face_distance(face_encodings, face_to_compare):
    """
    Given a list of face encodings, compare them to a known face encoding and get a euclidean distance
    for each comparison face. The distance tells you how similar the faces are.
    :param faces: List of face encodings to compare
    :param face_to_compare: A face encoding to compare against
    :return: A numpy ndarray with the distance for each face in the same order as the 'faces' array
    """
    import numpy as np
    if len(face_encodings) == 0:
        return np.empty((0))

    #return 1/np.linalg.norm(face_encodings - face_to_compare, axis=1)
    return np.sum(face_encodings*face_to_compare,axis=1)

# def load_model(model_dir, meta_file, ckpt_file):
#     model_dir_exp = os.path.expanduser(model_dir)
#     saver = tf.train.import_meta_graph(os.path.join(model_dir_exp, meta_file))
#     saver.restore(tf.get_default_session(), os.path.join(model_dir_exp, ckpt_file))

def _chinese_whispers(encoding_list, threshold=0.55, iterations=20):
    """ Chinese Whispers Algorithm

    Modified from Alex Loveless' implementation,
    http://alexloveless.co.uk/data/chinese-whispers-graph-clustering-in-python/

    Inputs:
        encoding_list: a list of facial encodings from face_recognition
        threshold: facial match threshold,default 0.6
        iterations: since chinese whispers is an iterative algorithm, number of times to iterate

    Outputs:
        sorted_clusters: a list of clusters, a cluster being a list of imagepaths,
            sorted by largest cluster to smallest
    """

    #from face_recognition.api import _face_distance
    from random import shuffle
    import networkx as nx
    # Create graph
    nodes = []
    edges = []

    image_paths, encodings = zip(*encoding_list)

    if len(encodings) <= 1:
        print ("No enough encodings to cluster!")
        return []

    for idx, face_encoding_to_check in enumerate(encodings):
        # Adding node of facial encoding
        node_id = idx+1

        # Initialize 'cluster' to unique value (cluster of itself)
        node = (node_id, {'cluster': image_paths[idx], 'path': image_paths[idx]})
        nodes.append(node)

        # Facial encodings to compare
        if (idx+1) >= len(encodings):
            # Node is last element, don't create edge
            break

        compare_encodings = encodings[idx+1:]
        distances = face_distance(compare_encodings, face_encoding_to_check)
        encoding_edges = []
        for i, distance in enumerate(distances):
            if distance > threshold:
                # Add edge if facial match
                edge_id = idx+i+2
                encoding_edges.append((node_id, edge_id, {'weight': distance}))

        edges = edges + encoding_edges

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # Iterate
    for _ in range(0, iterations):
        cluster_nodes = G.nodes()
        shuffle(cluster_nodes)
        for node in cluster_nodes:
            neighbors = G[node]
            clusters = {}

            for ne in neighbors:
                if isinstance(ne, int):
                    if G.node[ne]['cluster'] in clusters:
                        clusters[G.node[ne]['cluster']] += G[node][ne]['weight']
                    else:
                        clusters[G.node[ne]['cluster']] = G[node][ne]['weight']

            # find the class with the highest edge weight sum
            edge_weight_sum = 0
            max_cluster = 0
            #use the max sum of neighbor weights class as current node's class
            for cluster in clusters:
                if clusters[cluster] > edge_weight_sum:
                    edge_weight_sum = clusters[cluster]
                    max_cluster = cluster

            # set the class of target node to the winning local class
            G.node[node]['cluster'] = max_cluster

    clusters = {}

    # Prepare cluster output
    for (_, data) in G.node.items():
        cluster = data['cluster']
        path = data['path']

        if cluster:
            if cluster not in clusters:
                clusters[cluster] = []
            clusters[cluster].append(path)

    # Sort cluster output
    sorted_clusters = sorted(clusters.values(), key=len, reverse=True)

    return sorted_clusters

def cluster_facial_encodings(facial_encodings):
    """ Cluster facial encodings

        Intended to be an optional switch for different clustering algorithms, as of right now
        only chinese whispers is available.

        Input:
            facial_encodings: (image_path, facial_encoding) dictionary of facial encodings

        Output:
            sorted_clusters: a list of clusters, a cluster being a list of imagepaths,
                sorted by largest cluster to smallest

    """

    if len(facial_encodings) <= 1:
        print ("Number of facial encodings must be greater than one, can't cluster")
        return []

    # Only use the chinese whispers algorithm for now
    sorted_clusters = _chinese_whispers(facial_encodings.items())
    return sorted_clusters

# def compute_facial_encodings(sess,images_placeholder,embeddings,phase_train_placeholder,image_size,
#                     embedding_size,nrof_images,nrof_batches,emb_array,batch_size,paths):
#     """ Compute Facial Encodings

#         Given a set of images, compute the facial encodings of each face detected in the images and
#         return them. If no faces, or more than one face found, return nothing for that image.

#         Inputs:
#             image_paths: a list of image paths

#         Outputs:
#             facial_encodings: (image_path, facial_encoding) dictionary of facial encodings

#     """

#     for i in range(nrof_batches):
#         start_index = i*batch_size
#         end_index = min((i+1)*batch_size, nrof_images)
#         paths_batch = paths[start_index:end_index]
#         images = facenet.load_data(paths_batch, False, False, image_size)
#         feed_dict = { images_placeholder:images, phase_train_placeholder:False }
#         emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)

#     facial_encodings = {}
#     for x in range(nrof_images):
#         facial_encodings[paths[x]] = emb_array[x,:]


#     return facial_encodings

def get_onedir(paths):
    dataset = []
    path_exp = os.path.expanduser(paths)
    if os.path.isdir(path_exp):
        images = os.listdir(path_exp)
        image_paths = [os.path.join(path_exp,img) for img in images]

        for x in image_paths:
            if os.path.getsize(x)>0:
                dataset.append(x)
        
    return dataset 


# def main(args):
#     """ Main

#     Given a list of images, save out facial encoding data files and copy
#     images into folders of face clusters.

#     """
#     from os.path import join, basename, exists
#     from os import makedirs
#     import numpy as np
#     import shutil
#     import sys

#     if not exists(args.output):
#         makedirs(args.output)

#     with tf.Graph().as_default():
#         with tf.Session() as sess:
#             image_paths = get_onedir(args.input)
#             #image_list, label_list = facenet.get_image_paths_and_labels(train_set)

#             meta_file, ckpt_file = facenet.get_model_filenames(os.path.expanduser(args.model_dir))
            
#             print('Metagraph file: %s' % meta_file)
#             print('Checkpoint file: %s' % ckpt_file)
#             load_model(args.model_dir, meta_file, ckpt_file)
            
#             # Get input and output tensors
#             images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
#             embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
#             phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            
#             image_size = images_placeholder.get_shape()[1]
#             print("image_size:",image_size)
#             embedding_size = embeddings.get_shape()[1]
        
#             # Run forward pass to calculate embeddings
#             print('Runnning forward pass on images') 

#             nrof_images = len(image_paths)
#             nrof_batches = int(math.ceil(1.0*nrof_images / args.batch_size))
#             emb_array = np.zeros((nrof_images, embedding_size))
#             facial_encodings = compute_facial_encodings(sess,images_placeholder,embeddings,phase_train_placeholder,image_size,
#                 embedding_size,nrof_images,nrof_batches,emb_array,args.batch_size,image_paths)
#             sorted_clusters = cluster_facial_encodings(facial_encodings)
#             num_cluster = len(sorted_clusters)
                
#             # Copy image files to cluster folders
#             for idx, cluster in enumerate(sorted_clusters):
#                 #save all the cluster
#                 cluster_dir = join(args.output, str(idx))
#                 if not exists(cluster_dir):
#                     makedirs(cluster_dir)
#                 for path in cluster:
#                     shutil.copy(path, join(cluster_dir, basename(path)))

# def parse_args():
#     """Parse input arguments."""
#     import argparse
#     parser = argparse.ArgumentParser(description='Get a shape mesh (t-pose)')
#     parser.add_argument('--model_dir', type=str, help='model dir', required=True)
#     parser.add_argument('--batch_size', type=int, help='batch size', required=30)
#     parser.add_argument('--input', type=str, help='Input dir of images', required=True)
#     parser.add_argument('--output', type=str, help='Output dir of clusters', required=True)
#     args = parser.parse_args()

#     return args

# if __name__ == '__main__':
#     """ Entry point """
#     main(parse_args())