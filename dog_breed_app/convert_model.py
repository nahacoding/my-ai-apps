"""Orange3 모델(.pkcls) -> 가볍고 이식성 좋은 가중치 파일(.npz) 변환 스크립트.

모델을 다시 학습했거나 다른 .pkcls를 쓰고 싶을 때만 실행하면 됩니다.
Orange3가 설치된 파이썬 환경에서 실행하세요 (앱 실행 자체에는 Orange가 필요 없습니다).

    python convert_model.py Dog_model.pkcls dog_model.npz
"""
import pickle
import sys

import numpy as np


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def main(src="Dog_model.pkcls", dst="dog_model.npz"):
    with open(src, "rb") as f:
        model = pickle.load(f)

    sk = model.skl_model  # Orange 로지스틱 회귀 내부의 scikit-learn 모델
    X = model.instances.X  # 모델에 저장된 학습용 임베딩
    coef, intercept = sk.coef_, sk.intercept_

    # 변환이 정확한지 검증: Orange 예측과 numpy 예측이 같아야 합니다.
    diff = np.abs(softmax(X @ coef.T + intercept) - model(X, model.Probs)).max()
    assert diff < 1e-6, f"변환 결과가 Orange와 다릅니다 (최대 오차 {diff})"

    np.savez_compressed(
        dst,
        coef=coef.astype(np.float32),
        intercept=intercept.astype(np.float32),
        classes=np.array(model.domain.class_var.values),
        n_train=np.array(X.shape[0]),
        train_emb_mean=np.array(X.mean(), dtype=np.float32),
    )
    print(f"저장 완료: {dst} (클래스 {len(coef)}개, 특징 {coef.shape[1]}차원, 최대 오차 {diff:.1e})")


if __name__ == "__main__":
    main(*sys.argv[1:3])
