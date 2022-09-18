import tkinter as tk
from tkinter import *
from PIL import Image
from PIL import ImageTk
import tqdm
import os
import imageio
if 0:
    root = tk.Tk()
    root.geometry("200x200")
# '0001_American_Beauty\res_60_margin_40_eps_0.28_KNN_5'  
path_mdf = '/home/hanoch/results/face_reid/face_net/0001_American_Beauty/res_60_margin_40_eps_0.28_KNN_5/re_id'
filenames = [os.path.join(path_mdf, x) for x in os.listdir(path_mdf)
                    if x.endswith('png') or x.endswith('jpg')]
if not bool(filenames):
    raise ValueError('No files at that folder')

format = 'GIF-FI'
# format ='GIF'
images = []

for file_inx, file in enumerate(tqdm.tqdm(sorted(filenames))):
    if 1:
        img = Image.open(file)
        img.thumbnail((int(img.size[0]/4), int(img.size[1]/4)), Image.ANTIALIAS)
    else:
        imio = imageio.imread(file)
        img = Image.fromarray(imio).resize((int(imio.shape[0]/4), int(imio.shape[1]/4)))
    images.append(img)
    # images.append(imageio.imread(file))
# imageio.mimsave(os.path.join(path_mdf, str(path_mdf.split('/')[-2]) + '_movie.gif'), images, format=format, duration=len(images)/300)
images[0].save(os.path.join(path_mdf, str(path_mdf.split('/')[-2]) + "thumbnail.gif"), format='GIF', append_images=images,
        save_all=True, duration=1700, loop=0)

# writer = imageio.get_writer('test.mp4', fps = 30,
    # codec='mjpeg', quality=10, pixelformat='yuvj444p')

    # loading the images
    # img = ImageTk.PhotoImage(Image.open(file))
  
# l = Label()
# l.pack()




import glob
from PIL import Image
def make_gif(frame_folder):
    frames = [Image.open(image) for image in glob.glob(f"{frame_folder}/*.JPG")]
    frame_one = frames[0]
    frame_one.save("my_awesome.gif", format="GIF", append_images=frames,
               save_all=True, duration=100, loop=0)
    
if __name__ == "__main__":
    make_gif("/path/to/images")