import pandas as pd
import os
import numpy as np
import PIL.ImageColor as ImageColor
import matplotlib.pyplot as plt
import tqdm
import matplotlib
color_space = [ImageColor.getrgb(n) for n, c in ImageColor.colormap.items()][7::4] # avoid th aliceblue a light white one


# detection_path = '/home/hanoch/results/face_reid/face_net'
detection_path = '/media/results/face_reid'

pr_recall = pd.read_csv(os.path.join(detection_path, 'precision_recall.csv'), index_col=False)

plt.figure(figsize=(15, 14))
ix = 0
for reid_method in tqdm.tqdm(np.unique(pr_recall.reid_method)):
    df_reid_method = pr_recall[pr_recall.reid_method == reid_method]
    for min_cluster_size in np.unique(df_reid_method.min_cluster_size):
        df_min_cluster_size = df_reid_method[df_reid_method.min_cluster_size == min_cluster_size]
        for min_face_res in np.unique(df_min_cluster_size.min_face_res):
            df_min_face_res = df_min_cluster_size[df_min_cluster_size.min_face_res == min_face_res]
            if reid_method == 'dbscan':
                for dbscan_eps in np.unique(df_min_face_res.dbscan_eps):
                    df_dbscan_eps = df_min_face_res[df_min_face_res.dbscan_eps == dbscan_eps]
                    precision = np.mean(df_dbscan_eps.precision)
                    if len(df_dbscan_eps.precision) != np.unique(df_dbscan_eps.video_name).shape[0]:
                        print('Averaged over {} movies should be {}'.format(len(df_dbscan_eps.precision), np.unique(df_dbscan_eps.video_name)))
                    print(precision)
                    recall = np.mean(df_dbscan_eps.recall)
                    plt.plot(recall, precision, color=[x /256 for x in color_space[ix]], label=reid_method +
                    '_' + str(min_face_res) + '_' + str(min_cluster_size) + '_' + str(dbscan_eps.__format__('.3f')) + '_PR: ' + str(precision.__format__('.2f')) + '_' + str(recall.__format__('.2f')),
                         markersize=15, marker='*', linestyle='')
                    ix += 1
            elif reid_method == 'hdbscan':
                precision = np.mean(df_min_face_res.precision)
                recall = np.mean(df_min_face_res.recall)
                if len(df_dbscan_eps.precision) != np.unique(df_dbscan_eps.video_name).shape[0]:
                    print('Averaged over {} movies should be {}'.format(len(df_dbscan_eps.precision),
                                                                        np.unique(df_dbscan_eps.video_name)))
                print(precision)
                plt.plot(recall, precision, color=[x /256 for x in color_space[ix]], label=reid_method +
                                        '_' + str(min_face_res) + '_' + str(min_cluster_size) + '_PR: ' + str(precision.__format__('.2f')) + '_' + str(recall.__format__('.2f')),
                     markersize=15, marker='*', linestyle='')
                ix += 1
            else:
                raise ValueError("Invalid option")

plt.grid()
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Recall precision plot')
plt.legend(loc="lower right")

lw = 2
plt.plot([1, 0], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.savefig(os.path.join(detection_path, 'preciaion_recall.pdf'))
plt.savefig(os.path.join(detection_path, 'preciaion_recall.png'), format="png")

plt.show()
# plt.savefig(os.path.join(detection_path, 'preciaion_recall.png'), format="png")
# plt.savefig(os.path.join(detection_path, 'preciaion_recall.png'), format="png")
print('ka')
