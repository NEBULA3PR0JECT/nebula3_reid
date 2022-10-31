""""
To use:
  Add to ~/.pip/pip.conf
  [global]
  extra-index-url = http://74.82.29.209:8090
  trusted-host = http://74.82.29.209:8090
and then install expert:
pip install nebula3_experts==1.2.0
"""
import os
import sys
import warnings
import urllib
from facenet_pytorch.examples.reid_inference_mdf import FaceReId
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

# from movie.movie_db import MOVIE_DB  ; from __future__ import annotations

from experts.pipeline.api import *
from movie.movie_db import MOVIE_DB

movie_db = MOVIE_DB()
WEB_PREFIX = os.getenv('WEB_PREFIX', 'http://74.82.29.209:9000')
# from abc import ABC, abstractmethod

pilot = False #True  # False # till pipeline will be python3.8
# sys.path.insert(0, "/notebooks/")
# # sys.path.insert(0, './')
# sys.path.insert(0, 'nebula3_database/')
# sys.path.append("/notebooks/nebula3_database")
# from nebula3_database.config import NEBULA_CONF
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
#
# pipeline = PIPELINE()


# def get_movie(self, movie_id):
#     query = 'FOR doc IN Movies FILTER doc._id == "{}" RETURN doc'.format(movie_id)
#     cursor = self.db.aql.execute(query)
#     for data in cursor:
#         return data
#     return ({})



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


DEFAULT_FRAMES_PATH = "/tmp/movie_frames"

def download_image_file(image_url, image_location = None, remove_prev = True):
    """ download image file to location """
    result = False
    if image_location is None:
        image_location = os.path.join(DEFAULT_FRAMES_PATH, 'tmp.jpg')
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


    def process_movie(self, movie_id: str):  # "Movies/8367628636680745448"
        print(f'handling movie: {movie_id}')
        if 1:
            movie_db.get_movie(movie_id)
            list_mdfs = get_mdfs_path(movie_db=movie_db, movie_id=movie_id)
            mdfs_urls = [WEB_PREFIX + x for x in list_mdfs]

            movie_name = os.path.dirname(list_mdfs[0]).split('/')[-1]
            movie_id_no_database = movie_id.split('/')[-1]
            result_path = os.path.join(DEFAULT_FRAMES_PATH, movie_id_no_database, movie_name)

            if not os.path.exists(result_path):
                os.makedirs(result_path)

            download_res = [download_image_file(image_url=x, image_location=os.path.join(result_path, os.path.basename(y))) for x, y in zip(mdfs_urls, list_mdfs)]
            mdfs_local_paths = [os.path.join(result_path, os.path.basename(y)) for y in list_mdfs]
            print("MDFs downloaded from WEB server : {} ".format(all(download_res)))
            if not all(download_res):
                warnings.warn(
                    "MDF download from WEB server FAIL : exit!!!!")
                sys.exit()
        else:
            dict_tmp = eval(movie_id)
            if not pilot:
                list_mdfs = dict_tmp['mdfs_path']
        success = self.face_reid.reid_process_movie(mdfs_local_paths)
        # TODO write to DB like in https://github.com/NEBULA3PR0JECT/visual_clues/blob/ad9039ae3d3ee039a03acbba668bc316664359e5/run_visual_clues.py#L60
        # task actual work
        return success, None

    def get_name(self):
        return "re_id-task"

def test_pipeline_task(pipeline_id):
    task = MyTask()
    if pilot:
        pipeline = PipelineApi(None)
        pipeline.handle_pipeline_task(task, pipeline_id, stop_on_failure=True)
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
    if 0: # TODO uncomment
        if pipeline_id is None:
            warnings.warn(
                "PIPELINE_ID does not exist, exit!!!!")
            sys.exit()
    else:
        pipeline_id = '45f4739b-146a-4ae3-9d06-16dee5df6ca7'
    test_pipeline_task(pipeline_id)

""""

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