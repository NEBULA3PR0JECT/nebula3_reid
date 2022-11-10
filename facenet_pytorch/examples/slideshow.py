# import tkinter as tk
# from tkinter import *
# from PIL import ImageTk
import tqdm
import os
import glob
from pathlib import Path
import imageio
from argparse import ArgumentParser

if 0:
    root = tk.Tk()
    root.geometry("200x200")
# '0001_American_Beauty\res_60_margin_40_eps_0.28_KNN_5'  
# path_mdf = '/home/hanoch/results/face_reid/face_net/0001_American_Beauty/res_60_margin_40_eps_0.28_KNN_5/re_id'
# path = '/home/hanoch/results/face_reid/face_net/0001_American_Beauty'

def main():
    from PIL import Image

    parser = ArgumentParser()
    parser.add_argument("--path", type=str, help="re-id images folder",  required=True)
    args = parser.parse_args()
    print("*************   Slide show creation ************************ ")
    # path = '/home/hanoch/results/face_reid/face_net/3001_21_JUMP_STREET'
    # filenames = [os.path.join(path, x) for x in os.listdir(path)
    #                     if x.endswith('png') or x.endswith('jpg')]

    # filenames = glob.glob(path + '/**/*.jpg', recursive=True)
    #
    # if not bool(filenames):
    #     raise ValueError('No files at that folder')

    # format = 'GIF-FI'
    # format ='GIF'
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True  #https://stackoverflow.com/questions/12984426/pil-ioerror-image-file-truncated-with-big-images

    # for file_inx, file in enumerate(tqdm.tqdm(sorted(filenames))):
    for root, dirs, files in os.walk(args.path):
        for dir1 in dirs:
            if not os.path.isdir(os.path.join(root, dir1)):
                continue
            if 're_id' in os.listdir(os.path.join(root, dir1)) or 're_id' in os.listdir(root): # re_id folder are in current => no recursion of folders
                print("Process ", root, dirs)
                images = []
                re_id_path = [os.path.join(os.path.join(root, dir1), 're_id') if 're_id' in os.listdir(os.path.join(root, dir1)) else os.path.join(os.path.join(root), 're_id')][0]#os.path.join(os.path.join(root, dir1), 're_id')
                print(re_id_path)
                filenames = [os.path.join(re_id_path, x) for x in os.listdir(re_id_path)
                                    if x.endswith('png') or x.endswith('jpg')]
                for file in tqdm.tqdm(sorted(filenames)):
                    if 1:
                        img = Image.open(file)
                        img.thumbnail((int(img.size[0]/4), int(img.size[1]/4)), Image.ANTIALIAS)
                    else:
                        imio = imageio.imread(file)
                        img = Image.fromarray(imio).resize((int(imio.shape[0]/4), int(imio.shape[1]/4)))
                    images.append(img)
                path = Path(re_id_path)
                images[0].save(os.path.join(path.parent.parent.absolute(), str(root.split('/')[-1]) + '_' + str(dir1) + "_thumbnail.gif"), format='GIF',
                               append_images=images,
                               save_all=True, duration=2000, loop=0)
                print("Slide show saved {}".format(os.path.join(path.parent.parent.absolute(), str(root.split('/')[-1]) + '_' + str(dir1) + "_thumbnail.gif")))

            else:
                continue
            # images.append(imageio.imread(file))
        # imageio.mimsave(os.path.join(path_mdf, str(path_mdf.split('/')[-2]) + '_movie.gif'), images, format=format, duration=len(images)/300)

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

    # make_gif("/path/to/images")
    main()