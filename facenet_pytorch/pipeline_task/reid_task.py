""""
To use:
  Add to ~/.pip/pip.conf
  [global]
  extra-index-url = http://74.82.29.209:8090
  trusted-host = http://74.82.29.209:8090
and then install expert:
pip install nebula3_experts==1.2.3
"""
import os
import sys
import warnings
import urllib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from examples.reid_inference_mdf import FaceReId
from examples.remote_storage_utils import RemoteStorage
from experts.pipeline.api import *
from database.arangodb import NEBULA_DB
import re
nebula_db = NEBULA_DB()
# WEB_PREFIX = os.getenv('WEB_PREFIX', 'http://74.82.29.209:9000')
# DEFAULT_FRAMES_PATH = "/tmp/movie_frames"


remote_storage = RemoteStorage()
# from abc import ABC, abstractmethod
WEB_PATH_SAVE_REID = os.getenv('WEB_PATH_SAVE_REID', remote_storage.vp_config.WEB_PREFIX + '//datasets/media/services')#access ReID MDFs from Web address

pilot = True #True  # False # till pipeline will be python3.8
save_results_to_db = True
interleave_non_re_id_mdf = True
# Read the reId from web server
# http://74.82.29.209:9000//datasets/media/services/0001_American_Beauty/res_64_margin_40_eps_0.27_KNN_5/re_id/re-id_0001_American_Beauty_00.00.51.926-00.00.54.129_clipmdf_0034.jpg


def insert_json_to_db(combined_json, collection_name):
    """
    Inserts a JSON with global & local tokens to the database.
    """
    res = nebula_db.write_doc_by_key(combined_json, collection_name, overwrite=True, key_list=['movie_id'])

    print("Successfully inserted to database. Collection name: {}".format(collection_name))
    return res




def create_re_id_json(mdf_id_all, re_id_result_path, movie_name, web_path, movie_id):
    # from examples.remote_storage_utils import RemoteStorage

    # remote_storage = RemoteStorage()
    # from abc import ABC, abstractmethod
    # WEB_PATH_SAVE_REID = os.getenv('WEB_PATH_SAVE_REID',
    #                                remote_storage.vp_config.WEB_PREFIX + '//datasets/media/services')  # access ReID MDFs from Web address
    #
    # mdfs_local_dir = re_id_result_path
    # mdfs_web_dir = f'{remote_storage.vp_config.get_frames_path()}/{movie_name}'

    union_mdf_filenames = [os.path.join(re_id_result_path, x) for x in os.listdir(re_id_result_path)
                     if x.endswith('png') or x.endswith('jpg')]

    union_mdf_filenames.sort(key=lambda f: int(re.sub('\D', '', f)))

    urls = list()
    frames = []
    ix = 0
    for frame_number in union_mdf_filenames:
        base_file_name = os.path.basename(frame_number)

        v = None
        if 're-id' in base_file_name:
            v = mdf_id_all.get(base_file_name.split('re-id_')[-1], None)
            if not (v):
                print('could not copy re id image to web path {}'.format(base_file_name))

        frame_num = [int(re.sub('\D', '', base_file_name.split('.')[0])) if base_file_name.lower().endswith(
                                ('.png', '.jpg', '.jpeg')) else base_file_name][0]

        if v:
            mdf_has_id = False
            nested_dict = list()
            face_only_no_reid_dict = list()
            for reid in list(v.values()):
                if reid['id'] != - 1:
                    mdf_has_id = True
                    nested_dict.append({
                        'bbox': reid['bbox'].tolist(),
                        'id': reid['id'],
                        'prob': str(reid['prob'])
                    })
                else:
                    face_only_no_reid_dict.append({
                        'bbox': reid['bbox'].tolist(),
                        'prob': str(reid['prob'])
                    })

                    # url = remote_storage.vp_config.get_web_prefix() + os.path.join(mdfs_web_dir,
                    #                                                                os.path.basename(mdfs_local_dir),
                    #                                                                os.path.basename(frame_number))
            if mdf_has_id:
                frames.append({'frame_num': frame_num, "re-id": nested_dict, "face_no_id": face_only_no_reid_dict})
            # if face_only_no_reid_dict:
            #     frames.append({'frame_num': frame_num, "face_no_id": face_only_no_reid_dict})

            urls.append({'frame_num': frame_num, 'url': web_path[ix]})
            ix += 1
        else:
            urls.append({'frame_num': frame_num, 'url': web_path[ix]})
            ix += 1

    reid_json = {'movie_id': movie_id, 'frames': frames, 'urls': urls}
    return reid_json

def create_re_id_json_old(mdf_id_all, re_id_result_path, movie_name, web_path, movie_id):
    # from examples.remote_storage_utils import RemoteStorage

    # remote_storage = RemoteStorage()
    # from abc import ABC, abstractmethod
    # WEB_PATH_SAVE_REID = os.getenv('WEB_PATH_SAVE_REID',
    #                                remote_storage.vp_config.WEB_PREFIX + '//datasets/media/services')  # access ReID MDFs from Web address
    #
    # mdfs_local_dir = re_id_result_path
    # mdfs_web_dir = f'{remote_storage.vp_config.get_frames_path()}/{movie_name}'


    urls = list()
    frames = []
    ix = 0
    for frame_number, v in mdf_id_all.items():
        frame_number = [int(re.sub('\D', '', frame_number.split('.')[0])) if frame_number.lower().endswith(
                                ('.png', '.jpg', '.jpeg')) else frame_number][0]

        mdf_has_id = False
        nested_dict = list()
        for reid in list(v.values()):
            if reid['id'] != - 1:
                mdf_has_id = True
                nested_dict.append({
                    'bbox': reid['bbox'].tolist(),
                    'id': reid['id'],
                    'prob': str(reid['prob'])
                })
                # url = remote_storage.vp_config.get_web_prefix() + os.path.join(mdfs_web_dir,
                #                                                                os.path.basename(mdfs_local_dir),
                #                                                                os.path.basename(frame_number))
        if mdf_has_id:
            frames.append({'frame_num': frame_number, "re-id": nested_dict})
            urls.append({'frame_number': frame_number, 'url': web_path[ix]})
            ix += 1


    reid_json = {'movie_id': movie_id, 'frames': frames, 'urls': urls}
    return reid_json

# class PipelineTask(ABC):
#     @abstractmethod
#     def process_movie(self, movie_id: str):
#         """process movie and return True/False and error str if False
#         """
#         pass
#     @abstractmethod
#     def get_name(self):
#         pass

# T@@HK : TODO use get_mdfs_path() as a method from next movie_db release
def get_mdfs_path(movie_db, movie_id):
    query = 'FOR doc IN Movies FILTER doc._id == "{}" RETURN doc'.format(movie_id)
    cursor = movie_db.db.aql.execute(query)
    stages = []
    for movie in cursor:
        for mdf_path in movie['mdfs_path']:
            stages.append(mdf_path)
    return (stages)


def download_image_file(image_url, image_location=None, remove_prev=True):
    """ download image file to location """
    result = False
    if image_location is None:
        image_location = os.path.join(remote_storage.vp_config.LOCAL_FRAMES_PATH, 'tmp.jpg')
    # remove last file
    if remove_prev and os.path.exists(image_location):
        os.remove(image_location)

    url_link = image_url
    try:
        print(url_link)
        urllib.request.urlretrieve(url_link, image_location)
        result = True
    except Exception as e:
        print(e)
        print(f'An exception occurred while fetching {url_link}')
    return result, image_location

import subprocess
def merge_mdf_with_reid(mdfs_local_paths, re_id_result_path):
    re_id_files = os.listdir(re_id_result_path)
    if re_id_result_path:
        for mdf in mdfs_local_paths:
            if not any([os.path.splitext(os.path.basename(mdf))[0] in re_id_file for re_id_file in re_id_files]):
                frame_full_path = subprocess.getoutput('cp -p ' + mdf + ' ' + re_id_result_path)
                if frame_full_path != '':
                    warnings.warn('Could not copy MDF file to r_id temp folder')
                    print('Could not copy MDF {} file to r_id temp folder to {}'.format(mdf, re_id_result_path))
    else: # take the non-reid images which are actually the original MDF ones
        re_id_result_path = os.path.dirname(mdfs_local_paths[0])
        # warnings.warn('Could not copy MDF file to No r_id temp folder')
        # print('Could not copy MDF files to r_id temp folder to {}'.format(re_id_result_path))

    return re_id_result_path

class MyTask(PipelineTask):
    def __init__(self, *args, **kwargs):
        self.face_reid = FaceReId()
        # Modifying algo parameters N O T allowed !!! only post settings and only based upon ENV variable was set otherwise use the previous defaults
        self.face_reid.margin = os.getenv('REID_BB_MARGIN', self.face_reid.margin)
        self.face_reid.min_face_res = os.getenv('REID_BB_FACE_RES', self.face_reid.min_face_res)

        self.face_reid.re_id_method['cluster_threshold'] = os.getenv('REID_CLUSTER_THRESHOLD', self.face_reid.re_id_method['cluster_threshold'])
        self.face_reid.re_id_method['min_cluster_size'] = os.getenv('REID_CLUSTER_SIZE', self.face_reid.re_id_method['min_cluster_size'])
        self.cluster_threshold = self.face_reid.re_id_method['cluster_threshold']
        self.min_cluster_size = self.face_reid.re_id_method['min_cluster_size']

        self.collection_name = 's4_re_id'

    def restart(self):
        self.face_reid.re_id_method['cluster_threshold'] = self.cluster_threshold
        self.face_reid.re_id_method['min_cluster_size'] = self.min_cluster_size

    def process_movie(self, movie_id: str):  # "Movies/8367628636680745448"
        print(f'handling movie: {movie_id}')

        list_mdfs = nebula_db.get_movie_structure(movie_id)
        list_mdfs = list(list_mdfs.values())
        if not(list_mdfs):
            print("MDF list is empty movie_id : {} !!!" .format(movie_id))
            warnings.warn("MDF list is empty !!!")
        mdfs_urls = [remote_storage.vp_config.WEB_PREFIX + x for x in list_mdfs]

        movie_name = os.path.dirname(list_mdfs[0]).split('/')[-1]
        print("movie_name: {}".format(movie_name))
        movie_id_no_database = movie_id.split('/')[-1]
        result_path = os.path.join(remote_storage.vp_config.LOCAL_FRAMES_PATH, movie_id_no_database, movie_name)

        if not os.path.exists(result_path):
            os.makedirs(result_path)

        download_res = [download_image_file(image_url=x, image_location=os.path.join(result_path, os.path.basename(y))) for x, y in zip(mdfs_urls, list_mdfs)]
        mdfs_local_paths = [os.path.join(result_path, os.path.basename(y)) for y in list_mdfs]
        print("MDFs downloaded from WEB server : {} ".format(all(download_res)))
        if not all(download_res):
            warnings.warn(
                "MDF download from WEB server FAIL : exit!!!!")
            sys.exit()
        tmp_frame_path = os.path.join(remote_storage.vp_config.LOCAL_FRAMES_PATH_RESULTS_TO_UPLOAD, movie_name)
        re_id_image_file_web_path = WEB_PATH_SAVE_REID
        # Process ReId task
        self.restart()
        success, re_id_result_path, mdf_id_all = self.face_reid.reid_process_movie(path_mdf=mdfs_local_paths,
                                                                                   result_path_with_movie=tmp_frame_path,
                                                                                   save_results_to_db=True,
                                                                                   re_id_image_file_web_path=re_id_image_file_web_path)

        # re_id_result_path = re_id_result_path#f'{remote_storage.vp_config.get_local_frames_path()}/{movie_name}'
        re_id_mdfs_web_dir = f'{remote_storage.vp_config.get_frames_path()}/{movie_name}'
        re_id_mdfs_remote_web_dir = remote_storage.vp_config.get_web_prefix()
        if interleave_non_re_id_mdf:
            re_id_result_path = merge_mdf_with_reid(mdfs_local_paths, re_id_result_path)

        if re_id_result_path: # in case no faces no re_id images to copy at all
            web_path = remote_storage.save_re_id_mdf_to_web_n_create_remote_path(re_id_result_path,
                                                                                 re_id_mdfs_web_dir,
                                                                                 re_id_mdfs_remote_web_dir)

        if save_results_to_db and mdf_id_all:
            re_id_json = create_re_id_json(mdf_id_all, re_id_result_path, movie_name, web_path, movie_id)
        # elif interleave_non_re_id_mdf and not mdf_id_all: # in case mdf_id_all is empty and no ReID
        #     re_id_result_path = mdfs_local_paths[0]
        #     if re_id_result_path:
        #         if not isinstance(re_id_result_path, list):
        #             re_id_result_path = [re_id_result_path]
        #         urls = list()
        #         for ix, frame_number in enumerate(re_id_result_path):
        #             frame_number = os.path.basename(frame_number)
        #             frame_number = [int(re.sub('\D', '', frame_number.split('.')[0])) if frame_number.lower().endswith(
        #                 ('.png', '.jpg', '.jpeg')) else frame_number][0]
        #
        #             urls.append({'frame_number': frame_number, 'url': web_path[ix]})
        #             reid_json = {'movie_id': movie_id, 'urls': urls}

            insert_json_to_db(re_id_json, self.collection_name)
        return success, None

    def get_name(self):
        return "re_id_task"

    def process_movies(self, movie_ids: list, context: str):  # "Movies/8367628636680745448"
        for movie_id in movie_ids:
            self.process_movie(movie_id)
        return

def test_pipeline_task(pipeline_id):
    task = MyTask()
    if pilot:
        pipeline = PipelineApi(None)
        pipeline.handle_pipeline_task(task, pipeline_id, stop_on_failure=True)  # nebula_db.change_db('prodemo') pipeline.config.ARANGO_DB='prodemo' pipeline.movie_db.change_db('prodemo')
    else:
        task.process_movie('Movies/8367628636680745448')

if __name__ == '__main__': #'test':
    pipeline_id = os.getenv('PIPELINE_ID')
    if pipeline_id is None:
        warnings.warn(
            "PIPELINE_ID does not exist, exit!!!!")
        sys.exit()
    test_pipeline_task(pipeline_id)

""""
To run unitesting
In case running all workflow : go to nebula3_pipeline/sprint4.yaml replace the PIPELINE_ID with the one you got from ArangoDB
To debug an already pipeline ID 
remove the success field from the pipeine record "videoprocessing": "success" to "videoprocessing": ""
and 
"tasks": {} not "tasks": { "videoprocessing": {}}
NO!!!!! need to run but just modify the     pipeline_id = os.getenv('PIPELINE_ID')  to the relevant pipeline_id : # '45f4739b-146a-4ae3-9d06-16dee5df6ca7'#pipeline_id = '716074cf-605f-407d-8d33-c675805cce4a'
Run the following before pipeline.handle_pipeline_task() invoked in case want to change Db from ipc_200  ('ipc_200') if needed

nebula_db.change_db('prodemo')
pipeline.config.ARANGO_DB='prodemo'
pipeline.movie_db.change_db('prodemo')


hanoch@psw1a6ce7:~$ gradient workflows run --id cdc9127e-6c61-43b2-95fc-ba3ea1708950 --path nebula3_pipeline/sprint4.yaml

  "mdfs_path": [
    "/media/media/frames/2402585/frame0063.jpg",
    "/media/media/frames/2402585/frame0125.jpg",
    "/media/media/frames2402585/frame0346.jpg"
  ],


from nebula3-experts.experts.pipeline.api import *
def test_pipeline_task(pipeline_id):
    class MyTask(PipelineTask):
        def process_movie(self, movie_id: str) -> tuple[bool, str]:
            print (f'handling movie: {movie_id}')
            # task actual work
            return True, None
        def get_name(self) -> str:
            return "my-task"

    pipeline = PipelineApi(None)
    task = MyTask()
    pipeline.handle_pipeline_task(task, pipeline_id, stop_on_failure=True)
    
/media/media/frames/0001_American_Beauty    
"""
