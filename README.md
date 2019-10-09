# DSTC8_track2

dstc8 track2 official github site 입니다.
https://github.com/microsoft/dstc8-meta-dialog

## 환경설정

1. conda를 이용해 library를 인스톨 합니다.

~~~
  conda install dstc8-baseline python=3.7 cython
  conda activate dstc8-baseline
  conda install -c pytorch pytorch
  pip install -e .
  pip install -r requirements.txt
~~~

공식 사이트와 다른 점은, 1. pytorch=1.2.0을 사용, 2. requirements.txt를 추가적으로 다운로드한 것 입니다.

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

현재 ignore_domain.sh 만 안정적으로 돌아가는 상태입니다.
meta_learning 버전도 곧 추가 예정입니다. (valid loss 쪽 버그.)

~~~
  sh train_meta_learning.sh (meta-learning)
~~~

## log (tensorboard)

4. tensorboard 실행
~~~
  tensorboard --logdir=./metalwoz-retrieval-model_ignore_domain/logs
~~~

## predict

곧 업로드할 예정입니다.

## code 분석

(ignore-domain 기준) mldc/trainer/gpt_trainer.py에서 데이터 처리, training, validation, 모델 save, 모델 load 가 진행됩니다.

(meta-learning) class BatchPipePreparation 에서 meta-learning framework을 위한 데이터 전처리가 진행된 후, 
mldc/trainer/gpt_trainer.py에서 training, validation, 모델 save, 모델 load 가 진행됩니다.

각종 옵션들도(meta_lr, lr, ...) 디버깅 동안에는 mldc/trainer/gpt_trainer.py 에서 변수 지정 후 수동으로 변경 중입니다.

## ignore_domain

초반 몇 iteration 동안은 loss 가 감소하지만, 이후에는 lm_loss 기준 5.~~ 정도로 크게 변하지 않고 있습니다.
