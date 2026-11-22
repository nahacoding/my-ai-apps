import numpy as np
import streamlit as st

from dog_classifier import (
    DogModel,
    EmbeddingError,
    classify,
    display_name,
)

st.set_page_config(page_title="강아지 품종 분류기", page_icon="🐶", layout="centered")


@st.cache_resource
def load_model() -> DogModel:
    return DogModel()


@st.cache_data(show_spinner=False, max_entries=256)
def cached_classify(file_bytes: bytes) -> tuple[np.ndarray, float]:
    # 같은 사진을 다시 올리면 서버 호출 없이 캐시된 결과를 사용합니다.
    return classify(file_bytes, load_model())


model = load_model()

# ---------------------------------------------------------------- 사이드바
with st.sidebar:
    st.header("모델 정보")
    st.write(
        f"- 알고리즘: 로지스틱 회귀 (Orange3)\n"
        f"- 특징 추출: Inception v3 임베딩\n"
        f"- 학습 이미지: {model.n_train:,}장"
    )
    st.subheader("분류 가능한 품종")
    for label in model.classes:
        st.write(f"• {display_name(label)}")

# ---------------------------------------------------------------- 본문
st.title("🐶 강아지 품종 분류기")
st.write("강아지 사진을 올리면 10가지 품종 중 어떤 품종인지 예측해 드려요.")

tab_upload, tab_camera = st.tabs(["📁 사진 업로드", "📷 카메라 촬영"])
with tab_upload:
    uploaded = st.file_uploader(
        "강아지 사진을 선택하세요 (여러 장 가능)",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        accept_multiple_files=True,
    )
with tab_camera:
    shot = st.camera_input("강아지를 촬영해 보세요")

images = [(f.name, f.getvalue()) for f in uploaded]
if shot is not None:
    images.append(("camera.jpg", shot.getvalue()))

if not images:
    st.info("사진을 올리면 결과가 여기에 표시됩니다.")
    st.stop()

for name, data in images:
    st.divider()
    col_img, col_result = st.columns([1, 1.2], gap="large")
    col_img.image(data, caption=name, width="stretch")

    with col_result:
        try:
            with st.spinner("분석 중..."):
                proba, emb_mean = cached_classify(data)
        except ValueError as e:
            st.error(str(e))
            continue
        except EmbeddingError as e:
            st.error(
                f"임베딩 서버 호출에 실패했습니다: {e}\n\n"
                "잠시 후 다시 시도해 주세요. 계속 실패하면 Orange 임베딩 서버"
                "(api.garaza.io) 상태를 확인해야 합니다."
            )
            continue

        order = np.argsort(proba)[::-1]
        top = order[0]
        st.metric("예측 결과", display_name(model.classes[top]), f"{proba[top]:.1%}")

        if proba[top] < 0.5:
            st.warning("확신도가 낮아요. 강아지가 잘 보이는 다른 사진을 시도해 보세요.")

        st.write("**상위 5개 후보**")
        for i in order[:5]:
            st.progress(float(proba[i]), text=f"{display_name(model.classes[i])} · {proba[i]:.1%}")

        with st.expander("진단 정보"):
            st.write(
                f"임베딩 평균값: {emb_mean:.3f} "
                f"(학습 데이터 평균: {model.train_emb_mean:.3f})\n\n"
                "두 값이 크게 다르면 학습 때와 다른 임베딩이 사용되고 있다는 신호입니다."
            )

st.divider()
st.caption(
    "※ 이 모델은 10개 품종만 구분하므로, 강아지가 아닌 사진이나 목록에 없는 품종도 "
    "이 중 하나로 분류합니다."
)
