# Causal Inference Text Embedding Model

2025학년도 1학기 인과추론 수업 프로젝트에서, 텍스트 임베딩을 인과 추론에 적용한 논문 **Adapting Text Embeddings for Causal Inference** 를 기반으로 진행하였습니다. 본 프로젝트에서는 한국어 뉴스 기사 데이터를 활용하여, 기사에 대한 반응 수가 성별에 따라 유의미한 차이를 보이는지를 분석하고자 하였습니다.


---

## 개요

- 이 코드는 뉴스 기사 데이터를 활용하여 기자의 성별에 따른 반응 간의 인과 관계를 탐색하는 모델을 구성합니다.
- 주요 입력 데이터는 네이버 뉴스 기사로, 크롤링한 기사 원문 및 메타 정보(언론사, 기자명, 기자 홈페이지, 제목, 반응수, 댓글수, 작성시간 등)와 기자 홈페이지를 통해 추론한 **기자 성별**을 포함합니다.
- 기사 본문 정보를 임베딩한 후 성별에 따른 반응 수의 차이를 추정합니다.
- 최종적으로 처리를 받은 집단을 기준으로 평균 반응 효과(ATT)를 계산하여, 성별에 따른 유의미한 반응 차이가 존재하는지를 평가합니다. 
---

## 실행 방법

```bash
python main.py -c config/your_config.yaml
```


---

## 참고

[Adapting Text Embeddings for Causal Inference](https://arxiv.org/pdf/1905.12741)

[Causal-bert-pytorch](https://github.com/rpryzant/causal-bert-pytorch)

[KoBERT-Transformers](https://github.com/monologg/KoBERT-Transformers)