#!/bin/bash

echo 'fisrt'
# export CNN_WORKDIR=/home/hanoch/GIT/mahitl-aqua-fish-quality-cnn/modules
#!/bin/bash
echo "start runbatch.sh" >> ./scriptCE_sat.py.log
#python -u ./facenet_pytorch/examples/slideshow.py --path /home/hanoch/results/face_reid/face_net/0011_Gandhi >> ./scriptCE_sat.py.log </dev/null 2>&1
#python -u ./facenet_pytorch/examples/my_inference_mdf.py --cluster-threshold 0.28 --min-cluster-size 5 --min-face-res 64 --mtcnn-margin 40 --movie 0003_CASABLANCA >> ./scriptCE_sat.py.log </dev/null 2>&1
#python -u ./facenet_pytorch/examples/slideshow.py --path /home/hanoch/results/face_reid/face_net/0003_CASABLANCA >> ./scriptCE_sat.py.log </dev/null 2>&1
#python -u ./facenet_pytorch/examples/my_inference_mdf.py --cluster-threshold 0.28 --min-cluster-size 5 --min-face-res 64 --mtcnn-margin 40 --movie 0020_Raising_Arizona >> ./scriptCE_sat.py.log </dev/null 2>&1
#python -u ./facenet_pytorch/examples/slideshow.py --path /home/hanoch/results/face_reid/face_net/0020_Raising_Arizona >> ./scriptCE_sat.py.log </dev/null 2>&1
#python -u ./facenet_pytorch/examples/my_inference_mdf.py --cluster-threshold 0.28 --min-cluster-size 5 --min-face-res 64 --mtcnn-margin 40 --movie 0017_Pianist >> ./scriptCE_sat.py.log </dev/null 2>&1
#python -u ./facenet_pytorch/examples/slideshow.py --path /home/hanoch/results/face_reid/face_net/0017_Pianist >> ./scriptCE_sat.py.log </dev/null 2>&1
#python -u ./facenet_pytorch/examples/my_inference_mdf.py --task plot_id_over_mdf --cluster-threshold 0.3 --min-face-res 64 --min-cluster-size 5 --movie 0001_American_Beauty --mtcnn-margin 80 >> ./scriptCE_sat.py.log </dev/null 2>&1
#python -u ./facenet_pytorch/examples/my_inference_mdf.py --task plot_id_over_mdf --cluster-threshold 0.28 --min-face-res 64 --min-cluster-size 5 --movie 0001_American_Beauty --mtcnn-margin 80 >> ./scriptCE_sat.py.log </dev/null 2>&1
python -u ./facenet_pytorch/examples/slideshow.py --path /home/hanoch/results/face_reid/face_net/0001_American_Beauty >> ./scriptCE_sat.py.log </dev/null 2>&1

#python -u ./facenet_pytorch/examples/slideshow.py --path /home/hanoch/results/face_reid/face_net/0023_THE_BUTTERFLY_EFFECT >> ./scriptCE_sat.py.log </dev/null 2>&1
#python -u ./facenet_pytorch/examples/slideshow.py --path /home/hanoch/results/face_reid/face_net/0019_Pulp_Fiction >> ./scriptCE_sat.py.log </dev/null 2>&1
#python -u ./facenet_pytorch/examples/slideshow.py --path /home/hanoch/results/face_reid/face_net/0011_Gandhi >> ./scriptCE_sat.py.log </dev/null 2>&1


# python -u ./facenet_pytorch/examples/my_inference_mdf.py >> ./scriptCE_sat.py.log </dev/null 2>&1
#python -u ./facenet_pytorch/examples/my_inference_mdf.py --task plot_id_over_mdf --cluster-threshold 0.3 --min-face-res 60 --min-cluster-size 4 --movie 0001_American_Beauty --mtcnn-margin 60 >> ./scriptCE_sat.py.log </dev/null 2>&1
#python -u ./facenet_pytorch/examples/my_inference_mdf.py --task plot_id_over_mdf --cluster-threshold 0.3 --min-face-res 60 --min-cluster-size 6 --movie 0001_American_Beauty --mtcnn-margin 60 >> ./scriptCE_sat.py.log </dev/null 2>&1
#python -u ./facenet_pytorch/examples/my_inference_mdf.py --task plot_id_over_mdf --cluster-threshold 0.3 --min-face-res 60 --min-cluster-size 4 --movie 0001_American_Beauty --mtcnn-margin 20 >> ./scriptCE_sat.py.log </dev/null 2>&1
#python -u ./facenet_pytorch/examples/my_inference_mdf.py --task plot_id_over_mdf --cluster-threshold 0.3 --min-face-res 60 --min-cluster-size 5 --movie 0001_American_Beauty --mtcnn-margin 20 >> ./scriptCE_sat.py.log </dev/null 2>&1
#python -u ./facenet_pytorch/examples/my_inference_mdf.py --task plot_id_over_mdf --cluster-threshold 0.3 --min-face-res 60 --min-cluster-size 6 --movie 0001_American_Beauty --mtcnn-margin 20 >> ./scriptCE_sat.py.log </dev/null 2>&1
#python -u ./facenet_pytorch/examples/my_inference_mdf.py --cluster-threshold 0.28 --min-cluster-size 4 --min-face-res 64 --mtcnn-margin 40 --movie 0025_THE_LORD_OF_THE_RINGS_THE_RETURN_OF_THE_KING >> ./scriptCE_sat.py.log </dev/null 2>&1

#python -u ./facenet_pytorch/examples/my_inference_mdf.py --cluster-threshold 0.28 --min-cluster-size 5 --min-face-res 64 --mtcnn-margin 40 --movie 0025_THE_LORD_OF_THE_RINGS_THE_RETURN_OF_THE_KING >> ./scriptCE_sat.py.log </dev/null 2>&1
#python -u ./facenet_pytorch/examples/my_inference_mdf.py --cluster-threshold 0.28 --min-cluster-size 6 --min-face-res 64 --mtcnn-margin 40 --movie 0025_THE_LORD_OF_THE_RINGS_THE_RETURN_OF_THE_KING >> ./scriptCE_sat.py.log </dev/null 2>&1
#python -u ./facenet_pytorch/examples/my_inference_mdf.py --cluster-threshold 0.28 --min-cluster-size 5 --min-face-res 64 --mtcnn-margin 40 --movie 0023_THE_BUTTERFLY_EFFECT >> ./scriptCE_sat.py.log </dev/null 2>&1
#python -u ./facenet_pytorch/examples/my_inference_mdf.py --cluster-threshold 0.28 --min-cluster-size 6 --min-face-res 64 --mtcnn-margin 40 --movie 0023_THE_BUTTERFLY_EFFECT >> ./scriptCE_sat.py.log </dev/null 2>&1
#python -u ./facenet_pytorch/examples/my_inference_mdf.py --cluster-threshold 0.28 --min-cluster-size 5 --min-face-res 64 --mtcnn-margin 40 --movie 0019_Pulp_Fiction >> ./scriptCE_sat.py.log </dev/null 2>&1
#python -u ./facenet_pytorch/examples/my_inference_mdf.py --cluster-threshold 0.28 --min-cluster-size 6 --min-face-res 64 --mtcnn-margin 40 --movie 0019_Pulp_Fiction >> ./scriptCE_sat.py.log </dev/null 2>&1
#python -u ./facenet_pytorch/examples/my_inference_mdf.py --cluster-threshold 0.28 --min-cluster-size 5 --min-face-res 64 --mtcnn-margin 40 --movie 0011_Gandhi >> ./scriptCE_sat.py.log </dev/null 2>&1
#python -u ./facenet_pytorch/examples/my_inference_mdf.py --cluster-threshold 0.28 --min-cluster-size 6 --min-face-res 64 --mtcnn-margin 40 --movie 0011_Gandhi >> ./scriptCE_sat.py.log </dev/null 2>&1

#python -u ./facenet_pytorch/examples/my_inference_mdf.py --task plot_id_over_mdf --cluster-threshold 0.3 --min-face-res 60 --min-cluster-size 4 --movie 0001_American_Beauty --mtcnn-margin 60 >> ./scriptCE_sat.py.log </dev/null 2>&1
#python -u ./facenet_pytorch/examples/my_inference_mdf.py --task plot_id_over_mdf --cluster-threshold 0.3 --min-face-res 60 --min-cluster-size 6 --movie 0001_American_Beauty --mtcnn-margin 60 >> ./scriptCE_sat.py.log </dev/null 2>&1
#python -u ./facenet_pytorch/examples/my_inference_mdf.py --task plot_id_over_mdf --cluster-threshold 0.3 --min-face-res 60 --min-cluster-size 4 --movie 0001_American_Beauty --mtcnn-margin 20 >> ./scriptCE_sat.py.log </dev/null 2>&1
#python -u ./facenet_pytorch/examples/my_inference_mdf.py --task plot_id_over_mdf --cluster-threshold 0.3 --min-face-res 60 --min-cluster-size 5 --movie 0001_American_Beauty --mtcnn-margin 20 >> ./scriptCE_sat.py.log </dev/null 2>&1
#python -u ./facenet_pytorch/examples/my_inference_mdf.py --task plot_id_over_mdf --cluster-threshold 0.3 --min-face-res 60 --min-cluster-size 6 --movie 0001_American_Beauty --mtcnn-margin 20 >> ./scriptCE_sat.py.log </dev/null 2>&1

#python -u ./facenet_pytorch/examples/my_inference_mdf.py --cluster-threshold 0.28 --min-cluster-size 4 --min-face-res 64 --mtcnn-margin 20 --movie 0001_American_Beauty >> ./scriptCE_sat.py.log </dev/null 2>&1
#python -u ./facenet_pytorch/examples/my_inference_mdf.py --cluster-threshold 0.28 --min-cluster-size 5 --min-face-res 64 --mtcnn-margin 20 --movie 0001_American_Beauty >> ./scriptCE_sat.py.log </dev/null 2>&1
#python -u ./facenet_pytorch/examples/my_inference_mdf.py --cluster-threshold 0.28 --min-cluster-size 6 --min-face-res 64 --mtcnn-margin 20 --movie 0001_American_Beauty >> ./scriptCE_sat.py.log </dev/null 2>&1
#python -u ./facenet_pytorch/examples/my_inference_mdf.py --cluster-threshold 0.28 --min-cluster-size 7 --min-face-res 64 --mtcnn-margin 20 --movie 0001_American_Beauty >> ./scriptCE_sat.py.log </dev/null 2>&1
#python -u ./facenet_pytorch/examples/my_inference_mdf.py --cluster-threshold 0.3 --min-cluster-size 4 --min-face-res 64 --mtcnn-margin 20 --movie 0001_American_Beauty >> ./scriptCE_sat.py.log </dev/null 2>&1
#python -u ./facenet_pytorch/examples/my_inference_mdf.py --cluster-threshold 0.3 --min-cluster-size 5 --min-face-res 64 --mtcnn-margin 20 --movie 0001_American_Beauty >> ./scriptCE_sat.py.log </dev/null 2>&1
#python -u ./facenet_pytorch/examples/my_inference_mdf.py --cluster-threshold 0.3 --min-cluster-size 6 --min-face-res 64 --mtcnn-margin 20 --movie 0001_American_Beauty >> ./scriptCE_sat.py.log </dev/null 2>&1
#python -u ./facenet_pytorch/examples/my_inference_mdf.py --cluster-threshold 0.3 --min-cluster-size 7 --min-face-res 64 --mtcnn-margin 20 --movie 0001_American_Beauty >> ./scriptCE_sat.py.log </dev/null 2>&1

#python -u ./facenet_pytorch/examples/my_inference_mdf.py --cluster-threshold 0.3 --min-cluster-size 4 --min-face-res 60 --mtcnn-margin 60 --movie 0001_American_Beauty >> ./scriptCE_sat.py.log </dev/null 2>&1
#python -u ./facenet_pytorch/examples/my_inference_mdf.py --cluster-threshold 0.3 --min-cluster-size 5 --min-face-res 60 --mtcnn-margin 60 --movie 0001_American_Beauty >> ./scriptCE_sat.py.log </dev/null 2>&1
#python -u ./facenet_pytorch/examples/my_inference_mdf.py --cluster-threshold 0.3 --min-cluster-size 6 --min-face-res 60 --mtcnn-margin 60 --movie 0001_American_Beauty >> ./scriptCE_sat.py.log </dev/null 2>&1
#python -u ./facenet_pytorch/examples/my_inference_mdf.py --cluster-threshold 0.3 --min-cluster-size 7 --min-face-res 60 --mtcnn-margin 60 --movie 0001_American_Beauty >> ./scriptCE_sat.py.log </dev/null 2>&1

# python -u ./facenet_pytorch/examples/my_inference_mdf.py --cluster-threshold 0.25 --min-cluster-size 4 --min-face-res 60 --movie 0001_American_Beauty >> ./scriptCE_sat.py.log </dev/null 2>&1
# python -u ./facenet_pytorch/examples/my_inference_mdf.py --cluster-threshold 0.25 --min-cluster-size 5 --min-face-res 60 --movie 0001_American_Beauty >> ./scriptCE_sat.py.log </dev/null 2>&1
# python -u ./facenet_pytorch/examples/my_inference_mdf.py --cluster-threshold 0.25 --min-cluster-size 6 --min-face-res 60 --movie 0001_American_Beauty >> ./scriptCE_sat.py.log </dev/null 2>&1
# python -u ./facenet_pytorch/examples/my_inference_mdf.py --cluster-threshold 0.25 --min-cluster-size 7 --min-face-res 60 --movie 0001_American_Beauty >> ./scriptCE_sat.py.log </dev/null 2>&1
# python -u ./facenet_pytorch/examples/my_inference_mdf.py --cluster-threshold 0.25 --min-cluster-size 4 --min-face-res 72 --movie 0001_American_Beauty >> ./scriptCE_sat.py.log </dev/null 2>&1
# python -u ./facenet_pytorch/examples/my_inference_mdf.py --cluster-threshold 0.25 --min-cluster-size 5 --min-face-res 72 --movie 0001_American_Beauty >> ./scriptCE_sat.py.log </dev/null 2>&1
# python -u ./facenet_pytorch/examples/my_inference_mdf.py --cluster-threshold 0.25 --min-cluster-size 6 --min-face-res 72 --movie 0001_American_Beauty >> ./scriptCE_sat.py.log </dev/null 2>&1
# python -u ./facenet_pytorch/examples/my_inference_mdf.py --cluster-threshold 0.25 --min-cluster-size 7 --min-face-res 72 --movie 0001_American_Beauty >> ./scriptCE_sat.py.log </dev/null 2>&1

# python -u ./facenet_pytorch/examples/my_inference_mdf.py --cluster-threshold 0.40 >> ./scriptCE_sat.py.log </dev/null 2>&1
# python -u ./facenet_pytorch/examples/my_inference_mdf.py --cluster-threshold 0.45 >> ./scriptCE_sat.py.log </dev/null 2>&1
# python -u ./facenet_pytorch/examples/my_inference_mdf.py --cluster-threshold 0.35 --min-face-res 72 >> ./scriptCE_sat.py.log </dev/null 2>&1
# python -u ./facenet_pytorch/examples/my_inference_mdf.py --cluster-threshold 0.40 --min-face-res 72 >> ./scriptCE_sat.py.log </dev/null 2>&1
# python -u ./facenet_pytorch/examples/my_inference_mdf.py --cluster-threshold 0.45 --min-face-res 72 >> ./scriptCE_sat.py.log </dev/null 2>&1