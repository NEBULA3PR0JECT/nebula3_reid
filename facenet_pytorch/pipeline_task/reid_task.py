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
from movie.movie_db import MOVIE_DB

movie_db = MOVIE_DB()
# WEB_PREFIX = os.getenv('WEB_PREFIX', 'http://74.82.29.209:9000')
# DEFAULT_FRAMES_PATH = "/tmp/movie_frames"


remote_storage = RemoteStorage()
# from abc import ABC, abstractmethod
WEB_PATH_SAVE_REID = os.getenv('WEB_PATH_SAVE_REID', remote_storage.vp_config.WEB_PREFIX + '//datasets/media/services')#access ReID MDFs from Web address

pilot = True #True  # False # till pipeline will be python3.8
save_results_to_db = True
# Read the reId from web server
# http://74.82.29.209:9000//datasets/media/services/0001_American_Beauty/res_64_margin_40_eps_0.27_KNN_5/re_id/re-id_0001_American_Beauty_00.00.51.926-00.00.54.129_clipmdf_0034.jpg

# sys.path.insert(0, "/notebooks/")
# # sys.path.insert(0, './')
# sys.path.insert(0, 'nebula3_database/')
# sys.path.append("/notebooks/nebula3_database")
# from nebula3_database.config import NEBULA_CONF
# from nebula3_database.database.arangodb import DatabaseConnector
# from nebula3_database.database.arangodb import DatabaseConnector

# class PIPELINE:
#     def __init__(self):
#         self.config = NEBULA_CONF()
#         self.db_host = self.config.get_database_host()
#         self.database = self.config.get_playground_name()
#         self.gdb = DatabaseConnector()
#         self.db = self.gdb.connect_db(self.database)
#         self.nre = MOVIE_DB()
#         self.nre.change_db("prodemo")
#         self.db = self.nre.db
# pipeline = PIPELINE()

def insert_json_to_db(movie_db, combined_json):
    """
    Inserts a JSON with global & local tokens to the database.
    """

    movie_db.change_db("prodemo")
    db = movie_db

    query = 'UPSERT { movie_id: @movie_id } \
            INSERT { movie_id: @movie_id, frames: @frames, urls: @urls} \
            UPDATE { movie_id: @movie_id, frames: @frames , urls: @urls} \
            IN  s4_re_id'

    movie_db.db.aql.execute(query, bind_vars=combined_json)
    print("Successfully inserted to database.")

    return


def create_re_id_json(mdf_id_all, re_id_result_path, movie_name, web_path, movie_id):
    # from examples.remote_storage_utils import RemoteStorage

    # remote_storage = RemoteStorage()
    # from abc import ABC, abstractmethod
    WEB_PATH_SAVE_REID = os.getenv('WEB_PATH_SAVE_REID',
                                   remote_storage.vp_config.WEB_PREFIX + '//datasets/media/services')  # access ReID MDFs from Web address

    mdfs_local_dir = re_id_result_path
    mdfs_web_dir = f'{remote_storage.vp_config.get_frames_path()}/{movie_name}'
    urls = list()

    frames = []
    ix = 0
    for frame_number, v in mdf_id_all.items():
        mdf_has_id = False
        for reid in list(v.values()):
            if reid['id'] != - 1:
                mdf_has_id = True
                frames.append({
                    'frame_number': frame_number,
                    'bbox': reid['bbox'].tolist(),
                    'id': reid['id'],
                    'prob': str(reid['prob'])
                })
                # url = remote_storage.vp_config.get_web_prefix() + os.path.join(mdfs_web_dir,
                #                                                                os.path.basename(mdfs_local_dir),
                #                                                                os.path.basename(frame_number))
        if mdf_has_id:
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
    def restart(self):
        self.face_reid.re_id_method['cluster_threshold'] = self.cluster_threshold
        self.face_reid.re_id_method['min_cluster_size'] = self.min_cluster_size

    def process_movie(self, movie_id: str):  # "Movies/8367628636680745448"
        print(f'handling movie: {movie_id}')
        movie_db.get_movie(movie_id)
        list_mdfs = get_mdfs_path(movie_db=movie_db, movie_id=movie_id) # Arango DB Query
        if not(list_mdfs):
            print("MDF list is empty movie_id : {} !!!" .format(movie_id))
            warnings.warn("MDF list is empty !!!")
        mdfs_urls = [remote_storage.vp_config.WEB_PREFIX + x for x in list_mdfs]

        movie_name = os.path.dirname(list_mdfs[0]).split('/')[-1]
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

        mdfs_local_dir = re_id_result_path#f'{remote_storage.vp_config.get_local_frames_path()}/{movie_name}'
        mdfs_web_dir = f'{remote_storage.vp_config.get_frames_path()}/{movie_name}'
        remote_storage.save_re_id_mdf_to_web(mdfs_local_dir, mdfs_web_dir)

        mdf_filenames = [os.path.join(mdfs_local_dir, x) for x in os.listdir(mdfs_local_dir)
                     if x.endswith('png') or x.endswith('jpg')]

        web_path = list()
        for mdf_file in mdf_filenames:
            web_path.append(remote_storage.vp_config.get_web_prefix() + os.path.join(mdfs_web_dir, os.path.basename(mdfs_local_dir), os.path.basename(mdf_file)))
            print("Web path for ReID MDF: {}".format(web_path[-1]))
            # print(remote_storage.vp_config.get_web_prefix() + os.path.join(mdfs_web_dir, os.path.basename(mdfs_local_dir), os.path.basename(file)))
        # TODO write to DB like in https://github.com/NEBULA3PR0JECT/visual_clues/blob/ad9039ae3d3ee039a03acbba668bc316664359e5/run_visual_clues.py#L60
        # task actual work

        if save_results_to_db and mdf_id_all:
            re_id_json = create_re_id_json(mdf_id_all, re_id_result_path, movie_name, web_path, movie_id)
            insert_json_to_db(movie_db, re_id_json)
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
        pipeline.handle_pipeline_task(task, pipeline_id, stop_on_failure=True)  # HK to support movies read env param that process_by_context=True : a new parametet to the method => process_movies()
    else:
        task.process_movie('Movies/8367628636680745448')#('doc_movie_3132222071598952047')
# movie_id: "Movies/-3132222071598952047"

doc_movie_3132222071598952047 = {
  "name": "2402585",
  "url_path": "http://74.82.29.209:9000/datasets/media/movies/2402585.mp4",
  "orig_url": "http://74.82.29.209:9000/msrvtt/video8135.mp4",
  "scenes": [],
  "scene_elements": [
    [
      0,
      449
    ]
  ],
  "mdfs": [
    [
      346,
      125,
      63
    ]
  ],
  "mdfs_path": [
        "/media/media/frames/0001_American_Beauty/0001_American_Beauty_00.00.51.926-00.00.54.129_clipmdf_0034.jpg",
        "/media/media/frames/0001_American_Beauty/0001_American_Beauty_00.00.56.224-00.01.03.394_clipmdf_0050.jpg",
        "/media/media/frames/0001_American_Beauty/0001_American_Beauty_00.00.56.224-00.01.03.394_clipmdf_0124.jpg",
        "/media/media/frames/0001_American_Beauty/0001_American_Beauty_00.01.43.985-00.01.48.384_clipmdf_0097.jpg",
        "/media/media/frames/0001_American_Beauty/0001_American_Beauty_00.06.13.870-00.06.16.291_clipmdf_0045.jpg",
        "/media/media/frames/0001_American_Beauty/0001_American_Beauty_00.06.23.477-00.06.24.326_clipmdf_0039.jpg",
        "/media/media/frames/0001_American_Beauty/0001_American_Beauty_00.06.30.298-00.06.32.185_clipmdf_0011.jpg",
        "/media/media/frames/0001_American_Beauty/0001_American_Beauty_00.08.11.990-00.08.14.310_clipmdf_0045.jpg",


  ],
  "meta": {
    "fps": 29.97002997002997,
    "width": 320,
    "height": 240
  },
  "updates": 1,
  "source": "external"
}
if __name__ == '__main__': #'test':
    pipeline_id = os.getenv('PIPELINE_ID') # '45f4739b-146a-4ae3-9d06-16dee5df6ca7'
    if 1: # TODO uncomment
        if pipeline_id is None:
            warnings.warn(
                "PIPELINE_ID does not exist, exit!!!!")
            sys.exit()
    else:
        pipeline_id = '72160aa5-c096-4fb2-8874-92ffdedd028f'#'72160aa5-c096-4fb2-8874-92ffdedd028f'#'8fae7dfb-b091-47f3-81e4-d78ebaf844b3'#'45f4739b-146a-4ae3-9d06-16dee5df6ca7'
    test_pipeline_task(pipeline_id)

""""
To run unitesting
go to nebula3_pipeline/sprint4.yaml replace the PIPELINE_ID with the one you got from ArangoDB
remove the sucess field from the pipeine record "videoprocessing": "success" to "videoprocessing": ""
and 
"tasks": {} not "tasks": { "videoprocessing": {}}
NO!!!!! need to run but just modify the     pipeline_id = os.getenv('PIPELINE_ID')  to the relevant pipeline_id :
and movie_db.change_db('ipc_200') if needed
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
