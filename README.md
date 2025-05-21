Suppressive Dropout Experiment
==============================

구조적으로 억제항 기반 LRP 분해를 활용한 Suppressive Dropout 기법을 실험합니다.

디렉토리 구성:

- data/: CIFAR-10 데이터셋 저장 경로
- models/: 필요한 경우 모델 래퍼 정의
- strategy/: Dropout / Amplify 전략 정의
- utils/: 억제항 계산, Grad-CAM, 기타 유틸리티 함수
- train.py: 기본 학습 루프
- evaluate.py: 정확도 측정 루프
- run_all_experiments.py: 단일 전략 실험 실행
- results_logger.py: CSV 형태로 결과 저장

학습 설정:
- CIFAR-10 train set 중 5000개 subset 사용
- Batch size: 32
- 모델: ResNet-18
- 입력 크기: 64x64

현재 구현된 전략:
- Suppressive Channel Dropout
