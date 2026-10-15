"""Phase 1 데이터 확보.

Galaxy10 DECaLS(h5)를 받아 허블 분류 3종(타원/나선/불규칙)에서 각 1000장씩 뽑아
data/raw/에 png로 저장하고 data/labels.csv에 정답표를 기록한다.

사용법
    python src/01_download.py                 # zenodo에서 받아서 처리
    python src/01_download.py --h5 경로.h5     # 이미 받은 h5 파일 사용
    python src/01_download.py --from-folder   # data/raw/에 직접 넣은 사진으로 labels.csv 틀만 생성
"""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm

SEED = 42
PER_CLASS = 1000
URL = "https://zenodo.org/records/10845026/files/Galaxy10_DECals.h5"

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
H5_PATH = DATA / "Galaxy10_DECals.h5"
LABELS = DATA / "labels.csv"

GALAXY10_NAMES = [
    "Disturbed", "Merging", "Round Smooth", "In-between Round Smooth",
    "Cigar Shaped Smooth", "Barred Spiral", "Unbarred Tight Spiral",
    "Unbarred Loose Spiral", "Edge-on without Bulge", "Edge-on with Bulge",
]
# Galaxy10 번호 -> 탐구용 분류
HUBBLE = {
    2: "elliptical", 3: "elliptical", 4: "elliptical",
    5: "spiral", 6: "spiral", 7: "spiral", 8: "spiral", 9: "spiral",
    0: "irregular", 1: "irregular",
}
HUBBLE_KO = {"elliptical": "타원은하", "spiral": "나선은하", "irregular": "불규칙은하"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def download(url: str, dest: Path) -> None:
    """이어받기를 지원하는 다운로드. 이미 다 받았으면 건너뛴다."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        if r.status_code == 416:  # 이미 끝까지 받음
            part.rename(dest)
            return
        r.raise_for_status()
        if have and r.status_code != 206:  # 서버가 이어받기를 지원하지 않음
            have = 0
        total = int(r.headers.get("Content-Length", 0)) + have
        with open(part, "ab" if have else "wb") as f, tqdm(
            total=total, initial=have, unit="B", unit_scale=True, desc=dest.name
        ) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    part.rename(dest)


def sample_indices(labels: np.ndarray) -> dict[str, np.ndarray]:
    """허블 분류마다 PER_CLASS장씩 무작위로 뽑는다 (seed 42)."""
    rng = np.random.default_rng(SEED)
    hubble = np.array([HUBBLE[int(c)] for c in labels])
    picked = {}
    for name in ["elliptical", "spiral", "irregular"]:
        pool = np.flatnonzero(hubble == name)
        if len(pool) < PER_CLASS:
            sys.exit(f"{name} 후보가 {len(pool)}장뿐이라 {PER_CLASS}장을 뽑을 수 없다.")
        picked[name] = np.sort(rng.choice(pool, size=PER_CLASS, replace=False))
    return picked


def from_h5(h5_path: Path) -> None:
    with h5py.File(h5_path, "r") as f:
        labels = f["ans"][:]
        picked = sample_indices(labels)
        order = np.sort(np.concatenate(list(picked.values())))
        RAW.mkdir(parents=True, exist_ok=True)
        for old in RAW.glob("*.png"):
            old.unlink()
        rows = []
        for idx in tqdm(order, desc="png 저장"):
            img = f["images"][idx]
            cls = int(labels[idx])
            fname = f"g{idx:05d}.png"
            Image.fromarray(img).save(RAW / fname)
            rows.append({
                "filename": fname,
                "galaxy10_index": int(idx),
                "galaxy10_class": cls,
                "galaxy10_name": GALAXY10_NAMES[cls],
                "hubble_class": HUBBLE[cls],
                "hubble_class_ko": HUBBLE_KO[HUBBLE[cls]],
            })
    df = pd.DataFrame(rows)
    df.to_csv(LABELS, index=False, encoding="utf-8-sig")
    print(df["hubble_class_ko"].value_counts().to_string())


def from_folder() -> None:
    """다운로드가 막혔을 때. 사용자가 넣은 사진으로 labels.csv 틀을 만든다."""
    files = sorted(p.name for p in RAW.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not files:
        sys.exit(f"{RAW}에 사진이 없다. 사진을 넣고 다시 실행한다.")
    if LABELS.exists():
        sys.exit(f"{LABELS}가 이미 있다. 덮어쓰지 않으려고 멈춘다.")
    pd.DataFrame({
        "filename": files, "galaxy10_index": "", "galaxy10_class": "",
        "galaxy10_name": "", "hubble_class": "", "hubble_class_ko": "",
    }).to_csv(LABELS, index=False, encoding="utf-8-sig")
    print(f"사진 {len(files)}장으로 {LABELS} 틀을 만들었다. hubble_class를 아는 만큼 채운다.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", type=Path, help="이미 받은 Galaxy10_DECals.h5 경로")
    ap.add_argument("--from-folder", action="store_true", help="data/raw/의 사진을 그대로 사용")
    args = ap.parse_args()

    if args.from_folder:
        from_folder()
        return
    h5_path = args.h5 or H5_PATH
    if not h5_path.exists():
        print(f"{URL} 에서 받는다 (약 2.5GB).")
        try:
            download(URL, h5_path)
        except requests.RequestException as e:
            sys.exit(
                f"다운로드 실패: {e.__class__.__name__}\n"
                "네트워크에서 zenodo.org가 막혔을 수 있다. 다른 곳에서 받은 h5는 --h5 로,\n"
                "직접 모은 사진은 data/raw/에 넣고 --from-folder 로 실행한다."
            )
    from_h5(h5_path)
    n_png = len(list(RAW.glob("*.png")))
    n_rows = len(pd.read_csv(LABELS))
    print(f"data/raw png {n_png}장, labels.csv {n_rows}행")


if __name__ == "__main__":
    main()
