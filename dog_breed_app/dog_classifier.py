"""강아지 품종 분류 핵심 로직 (Streamlit과 분리되어 단독 테스트 가능).

Orange3에서 만든 모델은 두 단계로 구성되어 있습니다.
  1) 이미지 -> Inception v3 임베딩(2048차원)   : Orange의 'Image Embedding' 위젯과 동일한 서버 사용
  2) 임베딩 -> 품종 확률                       : 로지스틱 회귀 (가중치는 dog_model.npz에 저장)

학습할 때와 똑같은 임베딩을 써야 예측이 맞기 때문에, 전처리와 서버 호출 방식을
Orange(orange3-imageanalytics)의 구현 그대로 따라 했습니다.
"""
from __future__ import annotations

import io
import random
import time
import uuid
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageFile, ImageOps

ImageFile.LOAD_TRUNCATED_IMAGES = True  # Orange와 동일: 일부 깨진 JPEG도 허용

MODEL_PATH = Path(__file__).parent / "dog_model.npz"
EMBEDDING_URL = "https://api.garaza.io/image/inception-v3"  # Orange 기본 임베딩 서버
IMAGE_SIZE = (299, 299)  # Inception v3 입력 크기
EMBEDDING_DIM = 2048

KOREAN_NAMES = {
    "Beagle": "비글",
    "Boxer": "복서",
    "Bulldog": "불독",
    "Dachshund": "닥스훈트",
    "German_Shepherd": "저먼 셰퍼드",
    "Golden_Retriever": "골든 리트리버",
    "Labrador_Retriever": "래브라도 리트리버",
    "Poodle": "푸들",
    "Rottweiler": "로트와일러",
    "Yorkshire_Terrier": "요크셔 테리어",
}


def display_name(label: str) -> str:
    """'Golden_Retriever' -> '골든 리트리버 (Golden Retriever)'"""
    return f"{KOREAN_NAMES.get(label, label)} ({label.replace('_', ' ')})"


class EmbeddingError(RuntimeError):
    """임베딩 서버 호출에 실패했을 때 발생."""


class DogModel:
    """dog_model.npz에 저장된 로지스틱 회귀 가중치로 추론 (Orange/scikit-learn 불필요)."""

    def __init__(self, path: Path = MODEL_PATH):
        data = np.load(path, allow_pickle=False)
        self.coef = data["coef"].astype(np.float64)  # (클래스 수, 2048)
        self.intercept = data["intercept"].astype(np.float64)
        self.classes = [str(c) for c in data["classes"]]
        self.n_train = int(data["n_train"])
        self.train_emb_mean = float(data["train_emb_mean"])

    def predict_proba(self, embedding: np.ndarray) -> np.ndarray:
        z = self.coef @ np.asarray(embedding, dtype=np.float64) + self.intercept
        z -= z.max()  # 수치 안정성
        e = np.exp(z)
        return e / e.sum()


def prepare_jpeg(file_bytes: bytes) -> bytes:
    """업로드된 이미지를 Orange와 같은 방식으로 전처리해 JPEG 바이트로 변환.

    RGB 변환 -> 299x299 LANCZOS 리사이즈 -> JPEG 인코딩.
    (휴대폰 사진이 누워 보이지 않도록 EXIF 회전만 추가로 보정합니다.)
    """
    try:
        image = Image.open(io.BytesIO(file_bytes))
        image = ImageOps.exif_transpose(image)
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = image.resize(IMAGE_SIZE, Image.LANCZOS)
    except Exception as exc:  # PIL은 다양한 예외를 던지므로 한데 묶어 처리
        raise ValueError("이미지를 열 수 없습니다. 다른 파일을 시도해 주세요.") from exc

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def embed_image(jpeg_bytes: bytes, retries: int = 3, timeout: float = 60) -> np.ndarray:
    """Orange 임베딩 서버에 이미지를 보내 Inception v3 임베딩(2048차원)을 받아옵니다."""
    params = {
        "machine": str(uuid.getnode()),
        "session": str(random.randint(1, 10**10)),
    }
    headers = {"Content-Type": "image/jpeg"}
    last_error = "알 수 없는 오류"

    for attempt in range(1, retries + 1):
        try:
            response = requests.post(
                EMBEDDING_URL,
                params={**params, "retry": attempt},
                data=jpeg_bytes,
                headers=headers,
                timeout=timeout,
            )
            embedding = response.json().get("embedding") if response.content else None
            if embedding is not None and len(embedding) == EMBEDDING_DIM:
                return np.asarray(embedding, dtype=np.float64)
            last_error = f"서버 응답이 올바르지 않습니다 (HTTP {response.status_code})"
        except requests.RequestException as exc:
            last_error = f"서버에 연결하지 못했습니다 ({type(exc).__name__})"
        except ValueError:  # JSON 파싱 실패
            last_error = "서버 응답을 해석하지 못했습니다"
        if attempt < retries:
            time.sleep(1.5 * attempt)

    raise EmbeddingError(last_error)


def classify(file_bytes: bytes, model: DogModel) -> tuple[np.ndarray, float]:
    """이미지 바이트 -> (클래스별 확률, 임베딩 평균값)."""
    embedding = embed_image(prepare_jpeg(file_bytes))
    return model.predict_proba(embedding), float(embedding.mean())
