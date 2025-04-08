# [3주차] 심화과제 - Pre-trained 모델로 효율적인 NLP 모델 학습하기

#### Q1) 어떤 task를 선택하셨나요?

- MNLI task
- MNLI는 두 문장이 주어졌을 때 논리적으로 연결이 되어 있는지, 서로 모순되는지, 아니면 아예 무관한지 분류하는 문제이다.

#### Q2) 모델은 어떻게 설계하셨나요? 설계한 모델의 입력과 출력 형태가 어떻게 되나요?

- BERT는 sentence pair classification에 특화되어 있어 MNLI와 같이 두 문장 간의 복잡한 의미 관계를 파악하는데 효과적이다. BERT와 비슷한 성능을 가지면서 사이즈는 작은 DistilBERT를 사용하였다.
- 배치 사이즈는 데이터셋을 1000개만 사용하였기 때문에 8, 16, 32로 테스트해보고 overfitting이 되지 않고 성능이 잘 나오는 16으로 설정하였다.
- 모델의 default dropout(0.1)을 사용했을 때 overfitting이 보여, 0.3으로 설정하였다.
- 학습률을 3e-5에서는 loss curve가 불안정한 모습을 보여 2e-5로 설정하였다.
- epochs는 50으로 설정하고 대신 overfitting이 되지 않도록 early stopping을 적용하였다.

- 입력형태: torch.Size([1, 28])
- 출력형태: torch.Size([1, 3])

#### Q3) 실제로 pre-trained 모델을 fine-tuning했을 때 loss curve은 어떻게 그려지나요? 그리고 pre-train 하지 않은 Transformer를 학습했을 때와 어떤 차이가 있나요?

- pre-trined 모델을 사용하면 이미 자연어에 대한 학습이 되어 있기 때문에 초기 loss가 상대적으로 낮게 나온다.
  또한, loss curve가 더 빠르게 수렴하고 학습 속도도 더 빠른다.
- pre-trained 하지 않은 모델은 초기 loss가 높고, loss curve가 더 느리게 수렴하며 학습 속도도 더 느리다.
