# DSTC8_track2

dstc8 track2 official github site 입니다.
https://github.com/microsoft/dstc8-meta-dialog

## 환경설정

1. conda를 이용해 library를 인스톨 합니다.

공식 사이트와 다른 점은, 
  1. pytorch=1.2.0을 사용,
  2. requirements.txt를 추가적으로 다운로드합니다.

~~~
  conda install dstc8-baseline python=3.7 cython
  conda activate dstc8-baseline
  conda install -c pytorch pytorch
  pip install -e .
  pip install -r requirements.txt
~~~

2. symbolic link를 걸어줍니다. (용량이 100MB 넘음)

본 패스는 연구실 서버 기준입니다. 
~~~
   ln -s /ext2/jglee/DSTC8track2/pp-metalwoz-dir pp-metalwoz-dir
   ln -s /ext2/jglee/DSTC8track2/pp-reddit-dir pp-reddit-dir
~~~

3. training 코드를 실행합니다.

~~~
  sh
~~~
