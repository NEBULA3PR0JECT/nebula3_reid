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
from facenet_pytorch.examples.reid_inference_mdf import FaceReId
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

# from movie.movie_db import MOVIE_DB  ; from __future__ import annotations

import experts.pipeline.api
# from movie.movie_db import MOVIE_DB
from abc import ABC, abstractmethod

class PipelineTask(ABC):
    @abstractmethod
    def process_movie(self, movie_id: str):
        """process movie and return True/False and error str if False
        """
        pass
    @abstractmethod
    def get_name(self):
        pass

class MyTask(PipelineTask):
    def __init__(self, *args, **kwargs):
        self.face_reid = FaceReId()
        # Modifying algo parameters N O T allowed !!! only post settings and only based upon ENV variable was set otherwise use the previous defaults
        self.face_reid.margin = os.getenv('REID_BB_MARGIN', self.face_reid.margin)
        self.face_reid.min_face_res = os.getenv('REID_BB_FACE_RES', self.face_reid.min_face_res)

        self.face_reid.re_id_method['cluster_threshold'] = os.getenv('REID_CLUSTER_THRESHOLD', self.face_reid.re_id_method['cluster_threshold'])
        self.face_reid.re_id_method['min_cluster_size'] = os.getenv('REID_CLUSTER_SIZE', self.face_reid.re_id_method['min_cluster_size'])


    def process_movie(self, movie_id: str):
        print(f'handling movie: {movie_id}')
        dict_tmp = eval(movie_id)
        list_mdfs = dict_tmp['mdfs_path']
        success = self.face_reid.reid_process_movie(list_mdfs)
        # task actual work
        return success, None

    def get_name(self):
        return "re_id-task"

def test_pipeline_task(pipeline_id):
    task = MyTask()
    pilot = False # till pipeline will be python3.8
    if pilot:
        pipeline = PipelineApi(None)
        pipeline.handle_pipeline_task(task, pipeline_id, stop_on_failure=True)
    else:
        task.process_movie('doc_movie_3132222071598952047')
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
