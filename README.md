# Causal Inference Text Embedding Model

2025학년도 1학기 인과추론 수업 프로젝트로, 텍스트 임베딩을 인과 추론에 적용한 논문 **Adapting Text Embeddings for Causal Inference**를 기반으로, 한국어 기사 분석에 맞게 재구현되었습니다.

---

## 개요

- 이 코드는 뉴스 기사 데이터를 활용하여 기자의 성별에 따른 반응 간의 인과 관계를 탐색하는 모델을 구성합니다.
- 주요 입력 데이터는 네이버 뉴스 기사로, 크롤링한 기사 원문 및 메타 정보(언론사, 기자명, 기자 홈페이지, 제목, 반응수, 댓글수, 작성시간 등)와 기자 홈페이지를 통해 추론한 **기자 성별**을 포함합니다.
- 일반적인 BERT 임베딩보다, 인과 추론 목적에 맞게 조정한 임베딩이 ATT estimation 성능을 개선을 보이고자 했습니다.
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