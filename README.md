본 폴더는 GAN 기반의 GAIL 모델을 학습할 수 있다.

또한, 생성된 모델을 onnx 확장자로 변경할 수 있으며 성능을 plot 할 수 있는 코드도 제공한다.

1. VS code를 실행한다.
2. GAIL_try_1014.py를 열고 모든 dependency를 맞춘다.
3. 이 과정에서 데이터 셋을 수정할 수있으며, 본 연구에서는 1018.csv 파일을 이용한다.
4. 데이터 셋을 수정하거나 전처리를 할 수 있다.
5. 시계열 데이터 특성을 이해하기 위해 LSTM레이어를 이용하였다. 학습 후에 plot을 통해 확인할 수 있다.
6. 출력된 weight 값을  python -m tf2onnx.convert --saved-model policy_model_1 --output model_1108.onnx 명령어를 통해 onnx 파일로 변경한다.
7. 이 weight 파일을 https://github.com/mygummy/jahas1.9.4 의 resource 위치에 넣는다.
8. 위 링크의 jahas1.9.4를 실행하기 위한  Unity 프로그램을 설치한다.
9. 관련 dependency를 모두 맞춘다.
10. Unity 프로그램에서 실행할 수 있다.
