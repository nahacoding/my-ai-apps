# 🐶 강아지 품종 분류기

Orange3에서 학습한 모델을 Streamlit 웹앱으로 만든 프로젝트입니다.
강아지 사진을 올리면 10가지 품종(비글, 복서, 불독, 닥스훈트, 저먼 셰퍼드, 골든 리트리버,
래브라도 리트리버, 푸들, 로트와일러, 요크셔 테리어) 중 하나로 분류합니다.

## 파일 구성

| 파일 | 설명 |
|---|---|
| `app.py` | Streamlit 화면(UI) |
| `dog_classifier.py` | 이미지 전처리 · 임베딩 · 품종 예측 로직 |
| `dog_model.npz` | Orange 모델에서 추출한 로지스틱 회귀 가중치 (82KB) |
| `convert_model.py` | `.pkcls` → `.npz` 변환 스크립트 (모델을 새로 만들었을 때만 사용) |
| `requirements.txt` | 필요한 패키지 |

## 동작 방식

```
사진 → 299×299 리사이즈 → Orange 임베딩 서버(Inception v3) → 2048차원 벡터 → 로지스틱 회귀 → 품종 확률
```

학습할 때 Orange의 **Image Embedding** 위젯(Inception v3)을 썼기 때문에, 예측할 때도 같은
임베딩을 써야 합니다. 그래서 Orange와 같은 서버(`api.garaza.io`)를 같은 방식으로 호출합니다.
로지스틱 회귀 부분은 numpy로 직접 계산하므로 Orange·scikit-learn 설치가 필요 없습니다.

## 로컬 실행

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud 배포

1. 이 폴더의 파일을 GitHub 저장소에 올립니다.
2. [share.streamlit.io](https://share.streamlit.io)에서 저장소를 연결합니다.
3. Main file path를 `app.py`로 지정하고 Deploy 합니다.

## 알아둘 점

- **인터넷 연결 필요**: 임베딩을 Orange 서버에서 받아오므로, 서버가 점검 중이면 예측이 실패합니다.
- **개인정보**: 업로드한 사진(299×299로 축소된 이미지)이 Orange 임베딩 서버로 전송됩니다.
- **10개 품종만 구분**: 강아지가 아닌 사진이나 목록에 없는 품종도 10개 중 하나로 분류됩니다.
