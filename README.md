# DSTC8_track2

dstc8 track2 official github site 입니다.
https://github.com/microsoft/dstc8-meta-dialog

## 환경설정

0. git 을 다운로드 합니다.

~~~
 git clone https://github.com/jeonggwanlee/DSTC8_track2.git
 cd DSTC8_track2
~~~
***DSTC8_track2 에 들어가야지만, 이후 "pip install -e ." 가 제대로 작동합니다.
setup.py를 기준으로 library를 설치하기 때문입니다.***

1. conda를 이용해 library를 인스톨 합니다.

1.1 conda 버전을 확인 합니다.
  ~~~
    >> conda --version
    conda 4.7.10
  ~~~
~~~
  conda create -n dstc8-baseline python=3.7 cython
  conda activate dstc8-baseline (또는) source activate dstc8-baseline
  conda install -c pytorch pytorch
  pip install -e .
  pip install -r requirements.txt
~~~

공식 사이트와 다른 점은, 1. pytorch=1.2.0을 사용, 2. requirements.txt를 추가적으로 다운로드한 것 입니다.
(별도의 conflict는 발견되지 않았습니다.)

1.2 conda list 출력
~~~
(dstc8-baseline3) [jglee@storm DSTC8_track2]$ conda list
# packages in environment at /home/jglee/anaconda3/envs/dstc8-baseline3:
#
# Name                    Version                   Build  Channel
_libgcc_mutex             0.1                        main
absl-py                   0.8.1                    pypi_0    pypi
astor                     0.8.0                    pypi_0    pypi
attrs                     19.2.0                   pypi_0    pypi
blas                      1.0                         mkl
blis                      0.4.1                    pypi_0    pypi
boto3                     1.9.246                  pypi_0    pypi
botocore                  1.12.246                 pypi_0    pypi
ca-certificates           2019.8.28                     0
certifi                   2019.9.11                py37_0
cffi                      1.12.3           py37h2e261b9_0
chardet                   3.0.4                    pypi_0    pypi
click                     7.0                      pypi_0    pypi
cudatoolkit               10.0.130                      0
cymem                     2.0.2                    pypi_0    pypi
cython                    0.29.13          py37he6710b0_0
django                    2.2.6                    pypi_0    pypi
docutils                  0.15.2                   pypi_0    pypi
fairseq                   0.6.2                    pypi_0    pypi
fasttext                  0.8.3                    pypi_0    pypi
future                    0.18.0                   pypi_0    pypi
gast                      0.2.2                    pypi_0    pypi
google-pasta              0.1.7                    pypi_0    pypi
grpcio                    1.24.1                   pypi_0    pypi
h5py                      2.10.0                   pypi_0    pypi
hypothesis                3.88.3                   pypi_0    pypi
idna                      2.8                      pypi_0    pypi
intel-openmp              2019.4                      243
iterable-queue            1.2.2                    pypi_0    pypi
jmespath                  0.9.4                    pypi_0    pypi
joblib                    0.14.0                   pypi_0    pypi
keras-applications        1.0.8                    pypi_0    pypi
keras-preprocessing       1.1.0                    pypi_0    pypi
libedit                   3.1.20181209         hc058e9b_0
libffi                    3.2.1                hd88cf55_4
libgcc-ng                 9.1.0                hdf63c60_0
libgfortran-ng            7.3.0                hdf63c60_0
libstdcxx-ng              9.1.0                hdf63c60_0
markdown                  3.1.1                    pypi_0    pypi
mkl                       2019.4                      243
mkl-service               2.3.0            py37he904b0f_0
mkl_fft                   1.0.14           py37ha843d7b_0
mkl_random                1.1.0            py37hd6b4f25_0
mldc                      0.0.1                     dev_0    <develop>
murmurhash                1.0.2                    pypi_0    pypi
ncurses                   6.1                  he6710b0_1
ninja                     1.9.0            py37hfd86e86_0
nltk                      3.4.5                    pypi_0    pypi
numpy                     1.17.2           py37haad9e8e_0
numpy-base                1.17.2           py37hde5b4d6_0
onnx                      1.6.0                    pypi_0    pypi
openssl                   1.1.1d               h7b6447c_2
opt-einsum                3.1.0                    pypi_0    pypi
pandas                    0.25.1                   pypi_0    pypi
pip                       19.2.3                   py37_0
plac                      0.9.6                    pypi_0    pypi
portalocker               1.5.1                    pypi_0    pypi
preshed                   3.0.2                    pypi_0    pypi
protobuf                  3.10.0                   pypi_0    pypi
pycparser                 2.19                     py37_0
pydantic                  0.32.2                   pypi_0    pypi
pytext-nlp                0.1.2                    pypi_0    pypi
python                    3.7.4                h265db76_1
python-dateutil           2.8.0                    pypi_0    pypi
python-rapidjson          0.8.0                    pypi_0    pypi
pytorch                   1.2.0           py3.7_cuda10.0.130_cudnn7.6.2_0    pytorch
pytorch-pretrained-bert   0.6.2                    pypi_0    pypi
pytorch-transformers      1.0.0                    pypi_0    pypi
pytz                      2019.3                   pypi_0    pypi
readline                  7.0                  h7b6447c_5
regex                     2019.8.19                pypi_0    pypi
requests                  2.22.0                   pypi_0    pypi
runstats                  1.8.0                    pypi_0    pypi
s3transfer                0.2.1                    pypi_0    pypi
sacrebleu                 1.4.1                    pypi_0    pypi
scipy                     1.3.1                    pypi_0    pypi
sentencepiece             0.1.83                   pypi_0    pypi
setuptools                41.4.0                   py37_0
six                       1.12.0                   py37_0
spacy                     2.2.1                    pypi_0    pypi
sqlite                    3.30.0               h7b6447c_0
sqlparse                  0.3.0                    pypi_0    pypi
srsly                     0.1.0                    pypi_0    pypi
tabulate                  0.8.5                    pypi_0    pypi
tensorboard               2.0.0                    pypi_0    pypi
tensorboardx              1.9                      pypi_0    pypi
tensorflow-estimator      2.0.0                    pypi_0    pypi
tensorflow-gpu            2.0.0                    pypi_0    pypi
termcolor                 1.1.0                    pypi_0    pypi
thinc                     7.1.1                    pypi_0    pypi
tk                        8.6.8                hbc83047_0
torchtext                 0.4.0                    pypi_0    pypi
tqdm                      4.36.1                   pypi_0    pypi
typing                    3.7.4.1                  pypi_0    pypi
typing-extensions         3.7.4                    pypi_0    pypi
urllib3                   1.25.6                   pypi_0    pypi
wasabi                    0.2.2                    pypi_0    pypi
werkzeug                  0.16.0                   pypi_0    pypi
wheel                     0.33.6                   py37_0
wrapt                     1.11.2                   pypi_0    pypi
xz                        5.2.4                h14c3975_4
zlib                      1.2.11               h7b6447c_3
~~~


## symolic link 걸기

2. symbolic link를 걸어줍니다. (용량이 100MB 넘음)

본 패스는 연구실 서버 기준입니다. 
~~~
   ln -s /ext2/jglee/DSTC8track2/pp-metalwoz-dir pp-metalwoz-dir
   ln -s /ext2/jglee/DSTC8track2/pp-reddit-dir pp-reddit-dir
~~~

## training

3. training 코드를 실행합니다.

~~~
  sh train_ignore_domain.sh (도메인 무시하고 전체 넣어서 돌리기)
~~~

~~~
  sh train_meta_learning.sh (meta-learning)
~~~

## log (tensorboard)

4. tensorboard 실행
~~~
  tensorboard --logdir=./metalwoz-retrieval-model_ignore_domain/logs
~~~

## predict (== sentence generation)

곧 업로드할 예정입니다.

## code 분석

(ignore-domain 기준) mldc/trainer/gpt_trainer.py에서 데이터 처리, training, validation, 모델 save, 모델 load 가 진행됩니다.

(meta-learning) class BatchPipePreparation 에서 meta-learning framework을 위한 데이터 전처리가 진행된 후, 
mldc/trainer/gpt_trainer.py에서 training, validation, 모델 save, 모델 load 가 진행됩니다.

각종 옵션들도(meta_lr, lr, ...) 디버깅 동안에는 mldc/trainer/gpt_trainer.py 에서 변수 지정 후 수동으로 변경 중입니다.

## ignore_domain

초반 몇 iteration 동안은 loss 가 감소하지만, 이후에는 lm_loss 기준 5.~~ 정도로 크게 변하지 않고 있습니다.
