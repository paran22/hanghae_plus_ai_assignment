# [3주차] 기본과제 - DistilBERT로 뉴스 기사 분류 모델 학습하기

- [x] AG_News dataset 준비
  - Huggingface dataset의 `fancyzhx/ag_news`를 load
  - `collate_fn` 함수에 다음 수정사항들을 반영
    - Truncation과 관련된 부분들을 삭제
- [x] Classifier output, loss function, accuracy function 변경
  - 뉴스 기사 분류 문제는 binary classification이 아닌 일반적인 classification 문제입니다. MNIST 과제에서 했던 것 처럼 `nn.CrossEntropyLoss` 를 추가하고 `TextClassifier`의 출력 차원을 잘 조정하여 task를 풀 수 있도록 수정
  - 그리고 정확도를 재는 `accuracy` 함수도 classification에 맞춰 수정
- [x] 학습 결과 report
  - DistilBERT 실습과 같이 매 epoch 마다의 train loss를 출력하고 최종 모델의 test accuracy를 report 첨부

## 모델 학습 내용
### 데이터셋
- 뉴스데이터인 Huggingface dataset의 fancyzhx/ag_news를 사용한다.
- AG News는 World, Sports, Science, Business, Sci/Tech 5개의 카테고리로 분류된다.
### 모델 설계
- 텍스트를 분류하는데 강점이 있는 BERT를 pre-trained model로 선정하였고, BERT와 비슷한 성능을 가지면서 사이즈는 작은 DistillBERT를 사용하였다.
- optimizer로는 Adam보다 Transformer에 좋은 성능을 낸다고 알려진 AdamW를 사용하였다.

## 모델 학습 결과
- Train Accuracy: 0.8468 / Test Accuracy: 0.8605
![epoch 당 loss, train accuracy, test accuracy 변화 그래프](https://github.com/user-attachments/assets/e49a3abb-8b25-4c01-8b7a-6d10f7906751)


## 과제 링크
https://github.com/paran22/hanghae_plus_ai_assignment/blob/main/assignment/3-basic/3-basic.ipynb
