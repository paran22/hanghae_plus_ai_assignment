# [3주차] 심화과제 - Pre-trained 모델로 효율적인 NLP 모델 학습하기

### Q1) 어떤 task를 선택하셨나요?

- MNLI task
- MNLI는 두 문장이 주어졌을 때 논리적으로 연결이 되어 있는지, 서로 모순되는지, 아니면 아예 무관한지 분류하는 문제이다.

### Q2) 모델은 어떻게 설계하셨나요? 설계한 모델의 입력과 출력 형태가 어떻게 되나요?

#### 데이터셋

- Kaggle 데이터셋으로, 다음과 같이 구성되어 있다.
- premise: 기준이 되는 첫 번째 문장, hypothesis: 두 번째 문장
- label: 두 문장의 관계
  - entailment: 논리적 연결(0)
  - contradiction: 모순 관계(1)
  - neutral: 무관한 관계(2)

#### 입력과 출력 형태

- 입력형태: torch.Size([1, 28])
- 출력형태: torch.Size([1, 3])

#### 모델 설계 및 학습 결과

- BERT는 sentence pair classification에 특화되어 있어 MNLI와 같이 두 문장 간의 복잡한 의미 관계를 파악하는데 효과적이다.
- epochs는 50으로 설정하고 대신 overfitting이 되지 않도록 early stopping을 적용하였다. test accuracy가 일정 수준 이상 증가하지 않거나, train accuracy가 0.95 이상이면 학습을 중단하도록 하였다.
- optimizer로는 Adam보다 Transformer에 좋은 성능을 낸다고 알려진 AdamW를 사용하였다.

##### 1. DistilBERT

**모델 설계**

- BERT와 비슷한 성능을 가지면서 사이즈는 작은 DistilBERT를 사용하였다.
- 배치 사이즈는 데이터셋을 1000개만 사용하였기 때문에 8, 16, 32로 테스트해보고 overfitting이 되지 않고 성능이 잘 나오는 16으로 설정하였다.
- 모델의 default dropout(0.1)을 사용했을 때 overfitting이 보여, 0.3으로 설정하였다.
- 학습률을 3e-5에서는 loss curve가 불안정한 모습을 보여 2e-5로 설정하였다.
- 안정적인 학습을 위해 freeze layer를 추가하였다.

**학습 결과**

- fine tuning 전: Train Accuracy: 0.2870, Test Accuracy: 0.3200
- fine tuning 후: Train Accuracy: 0.5490(26.20% 향상), Test Accuracy: 0.3870(6.70% 향상)

![epoch 당 loss, train accuracy, test accuracy 변화 그래프](https://github.com/user-attachments/assets/fd772e1f-8630-4299-9091-532f79e6dd92)

**과제 링크**

https://github.com/paran22/hanghae_plus_ai_assignment/blob/main/assignment/3-hard/3-hard.ipynb

##### 2. RoBERTa

**모델 설계**

- DistillBERT에서 충분한 성능이 나오지 않아 더 큰 모델인 RoBERTa를 사용하였다.
- 과적합을 방지하기 위해 데이터셋 수를 5000으로 늘렸고, classifier_dropout을 0.3로, attention_probs_dropout_prob과 hidden_dropout_prob를 0.1로 설정하였다.
- warmup scheduler를 사용하여 학습률을 점진적으로 증가시켜 더 안정적으로 학습하도록 하였다.
- 메모리 사용량을 줄이고 학습 속도를 향상시키기 위해 Mixed Precision을 사용하고, Mixed Precision에서 발생할 수 있는 Gradient underflow를 방지하기 위해 Gradient Scaler를 적용하였다.
- freeze layer를 추가하였을때보다 추가하지 않았을 때 더 좋은 성능을 보였다.

**학습 결과**

- fine tuning 전: Train Accuracy: 0.3394, Test Accuracy: 0.3504
- fine tuning 후: Train Accuracy: 0.9630(62.36% 향상), Test Accuracy: 0.7812(43.08% 향상)
- freeze를 추가하였을 때는 test accuracy가 0.68정도

![epoch 당 loss, train accuracy, test accuracy 변화 그래프](https://github.com/user-attachments/assets/de80cbf6-07f0-46bf-a57d-f836a0543c85)

**과제 링크**
https://github.com/paran22/hanghae_plus_ai_assignment/blob/main/assignment/3-hard/3-hard-RoBERTa.ipynb

### Q3) 실제로 pre-trained 모델을 fine-tuning했을 때 loss curve은 어떻게 그려지나요? 그리고 pre-train 하지 않은 Transformer를 학습했을 때와 어떤 차이가 있나요?

- pre-trined 모델을 사용하면 이미 자연어에 대한 학습이 되어 있기 때문에 초기 loss가 상대적으로 낮게 나온다.
  또한, loss curve가 더 빠르게 수렴하고 학습 속도도 더 빠른다.
- pre-trained 하지 않은 모델은 초기 loss가 높고, loss curve가 더 느리게 수렴하며 학습 속도도 더 느리다.
