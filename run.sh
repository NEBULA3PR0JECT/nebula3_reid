#!/bin/bash

echo 'fisrt'
# export CNN_WORKDIR=/home/hanoch/GIT/mahitl-aqua-fish-quality-cnn/modules
#!/bin/bash
echo "start runbatch.sh" >> ./scriptCE_sat.py.log
# python -u ./facenet_pytorch/examples/my_inference_mdf.py >> ./scriptCE_sat.py.log </dev/null 2>&1
python -u ./facenet_pytorch/examples/my_inference_mdf.py --cluster-threshold 0.3 --min-cluster-size 4 --min-face-res 60 --mtcnn-margin 20 --movie 0001_American_Beauty >> ./scriptCE_sat.py.log </dev/null 2>&1
python -u ./facenet_pytorch/examples/my_inference_mdf.py --cluster-threshold 0.3 --min-cluster-size 5 --min-face-res 60 --mtcnn-margin 20 --movie 0001_American_Beauty >> ./scriptCE_sat.py.log </dev/null 2>&1
python -u ./facenet_pytorch/examples/my_inference_mdf.py --cluster-threshold 0.3 --min-cluster-size 6 --min-face-res 60 --mtcnn-margin 20 --movie 0001_American_Beauty >> ./scriptCE_sat.py.log </dev/null 2>&1
python -u ./facenet_pytorch/examples/my_inference_mdf.py --cluster-threshold 0.3 --min-cluster-size 7 --min-face-res 60 --mtcnn-margin 20 --movie 0001_American_Beauty >> ./scriptCE_sat.py.log </dev/null 2>&1

python -u ./facenet_pytorch/examples/my_inference_mdf.py --cluster-threshold 0.3 --min-cluster-size 4 --min-face-res 60 --mtcnn-margin 60 --movie 0001_American_Beauty >> ./scriptCE_sat.py.log </dev/null 2>&1
python -u ./facenet_pytorch/examples/my_inference_mdf.py --cluster-threshold 0.3 --min-cluster-size 5 --min-face-res 60 --mtcnn-margin 60 --movie 0001_American_Beauty >> ./scriptCE_sat.py.log </dev/null 2>&1
python -u ./facenet_pytorch/examples/my_inference_mdf.py --cluster-threshold 0.3 --min-cluster-size 6 --min-face-res 60 --mtcnn-margin 60 --movie 0001_American_Beauty >> ./scriptCE_sat.py.log </dev/null 2>&1
python -u ./facenet_pytorch/examples/my_inference_mdf.py --cluster-threshold 0.3 --min-cluster-size 7 --min-face-res 60 --mtcnn-margin 60 --movie 0001_American_Beauty >> ./scriptCE_sat.py.log </dev/null 2>&1

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