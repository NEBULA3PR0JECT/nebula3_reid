from os import getenv
import os
from paramiko import SSHClient, AutoAddPolicy
from scp import SCPClient


class VideoProcessingConf:
    def __init__(self) -> None:

        self.MOVIES_PATH = getenv('MOVIES_PATH', '/datasets/media/movies')
        self.LOCAL_MOVIES_PATH = getenv('LOCAL_MOVIES_PATH', '/tmp')
        self.FRAMES_PATH = getenv('FRAMES_PATH', '/datasets/media/frames')
        self.LOCAL_FRAMES_PATH = getenv('LOCAL_FRAMES_PATH', '/tmp/frames')
        self.LOCAL_FRAMES_PATH_RESULTS_TO_UPLOAD = getenv('LOCAL_FRAMES_PATH', '/tmp/frames/reid')
        self.WEB_PREFIX = getenv('WEB_PREFIX', 'http://74.82.29.209:9000')
        self.WEB_HOST = getenv('WEB_HOST', '74.82.29.209')
        self.WEB_USERPASS = getenv('WEB_USERPASS', 'paperspace:Nebula@12345')

    def get_movies_path(self):
        return (self.MOVIES_PATH)
    def get_local_movies_path(self):
        return (self.LOCAL_MOVIES_PATH)
    def get_frames_path(self):
        return (self.FRAMES_PATH)
    def get_local_frames_path(self):
        return (self.LOCAL_FRAMES_PATH)
    def get_web_prefix(self):
        return (self.WEB_PREFIX)
    def get_web_host(self):
        return (self.WEB_HOST)
    def get_web_userpass(self):
        return (self.WEB_USERPASS)


class RemoteStorage():
    def __init__(self):
        # super().__init__()
        self.vp_config = VideoProcessingConf()
        # make dirs
        if not os.path.isdir(self.vp_config.get_local_frames_path()):
            os.mkdir(self.vp_config.get_local_frames_path())
        # init ssh
        userpass = self.vp_config.get_web_userpass().split(":")
        self.ssh = SSHClient()
        self.ssh.set_missing_host_key_policy(AutoAddPolicy())
        # self.ssh.load_system_host_keys()
        self.ssh.connect(self.vp_config.get_web_host(),
                         username=userpass[0], password=userpass[1])
        self.scp = SCPClient(self.ssh.get_transport())
        # after init all

    def upload_files_to_web(self, uploads):
        try:
            for local_path, remote_path in uploads.items():
                self.scp.put(local_path, recursive=True, remote_path=remote_path)
        except Exception as exp:
            print(f'An exception occurred: {exp}')
            return False
        return True

    def save_re_id_mdf_to_web(self, mdfs_local_dir, mdfs_web_dir):
        # mdfs_web_dir = f'{self.vp_config.get_frames_path()}/{process_movie_param.movie_id}'
        # mdfs_local_dir = f'{self.vp_config.get_local_frames_path()}/{process_movie_param.movie_id}'
        uploads = {
            mdfs_local_dir: mdfs_web_dir
        }
        self.upload_files_to_web(uploads)