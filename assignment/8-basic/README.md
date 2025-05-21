## [8주차] 기본과제 - LoRA rank에 따른 학습 성능 비교해보기

- 모델: facebook/opt-350m
- 데이터셋: sahil2801/CodeAlpaca-20k

[과제코드](https://github.com/paran22/hanghae_plus_ai_assignment/blob/main/assignment/8-basic/8-basic.ipynb)

- LoRA rank에 따른 학습 성능 비교

| LoRA rank | train loss | train runtime | eval loss | eval runtime | Max Alloc |
| --------- | ---------- | ------------- | --------- | ------------ | --------- |
| 8         | 1.6888     | 3025.5256     | 1.56176   | 50.9211      | 3.1 GB    |
| 128       | 1.6778     | 3164.7489     | 1.55156   | 52.6994      | 3.2 GB    |
| 256       | 1.6777     | 3321.6138     | 1.55125   | 54.5512      | 3.4 GB    |

**train/loss**

https://api.wandb.ai/links/knospe1-gaeun/fbbgznur
![train_loss](https://github.com/user-attachments/assets/93a7b99a-2129-46da-a02c-65ef30ec7bf7)

**eval/loss**

https://api.wandb.ai/links/knospe1-gaeun/d3alfy60
![eval_loss](https://github.com/user-attachments/assets/f2093a82-dda5-4c79-9a0b-bb10264e53b9)

- rank를 증가시키면 학습 성능이 향상되는 것을 확인하였다. 그러나 128과 256의 경우 train loss와 eval loss가 비슷하게 유지되었다.
- rank를 증가시키면 runtime은 증가하고, 메모리 점유율도 증가하는 것을 확인하였다.
- rank에 따른 학습 성능의 차이를 고려하면 LoRA rank는 128로 설정하는 것이 적절해 보인다.
