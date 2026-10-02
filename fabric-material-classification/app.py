"""Glassmorphism Streamlit interface for fabric material recognition."""

from __future__ import annotations

from html import escape
from io import BytesIO
from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError

from src.predict import load_predictor, predict_image

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "models" / "best_model.pth"
MAX_IMAGE_BYTES = 12 * 1024 * 1024

st.set_page_config(
    page_title="Fabric Material AI",
    page_icon="🧵",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root { --ink:#f4f7ff; --muted:#aab7d8; --blue:#3b82f6; --cyan:#54d5ff; }
    html, body, [data-testid="stAppViewContainer"] {
      background: radial-gradient(ellipse at 50% 38%, rgba(24,88,255,.42), transparent 42%),
                  radial-gradient(ellipse at 6% 56%, rgba(30,103,255,.34), transparent 32%),
                  radial-gradient(ellipse at 96% 75%, rgba(0,113,255,.24), transparent 30%),
                  #070b18;
      color:var(--ink); font-family:Inter,'Segoe UI',sans-serif;
    }
    [data-testid="stHeader"] { background:transparent; }
    [data-testid="stToolbar"] { right:1rem; }
    .block-container { max-width:1180px; padding:1.5rem 2rem 3rem; }
    .brandbar { display:flex; align-items:center; justify-content:space-between; margin:0 0 3rem; }
    .brand { display:flex; gap:.75rem; align-items:center; font:700 1rem Inter,'Segoe UI',sans-serif; letter-spacing:.01em; }
    .brand-mark { display:grid; place-items:center; width:2.35rem; height:2.35rem; border-radius:13px;
      background:linear-gradient(145deg,#63d4ff,#1667e8 75%); box-shadow:0 5px 24px #1475ff77; font-size:1.15rem; }
    .brand small { display:block; color:#91a6d2; font:600 .65rem Inter,'Segoe UI',sans-serif; letter-spacing:.18em; margin-top:.12rem; }
    .online-pill { display:flex; align-items:center; gap:.5rem; border:1px solid #aec9ff24; background:#ffffff0b;
      border-radius:100px; padding:.55rem .9rem; color:#dce7ff; font-size:.76rem; backdrop-filter:blur(16px); }
    .online-dot { width:.45rem; height:.45rem; border-radius:50%; background:#58e2bb; box-shadow:0 0 10px #58e2bb; }
    .hero { text-align:center; margin:0 auto 2.2rem; max-width:760px; }
    .eyebrow { display:inline-flex; align-items:center; gap:.5rem; color:#a9d6ff; font-size:.7rem; font-weight:700;
      letter-spacing:.2em; text-transform:uppercase; border:1px solid #a4c8ff31; background:#ffffff09;
      padding:.45rem .8rem; border-radius:100px; }
    .hero h1 { color:#f7f9ff; font:800 clamp(2.1rem,5vw,3.8rem)/1.1 Inter,'Segoe UI',sans-serif; letter-spacing:-.055em; margin:1.1rem 0 .8rem; }
    .hero h1 span { background:linear-gradient(95deg,#75dcff,#5198ff 62%,#b2c8ff); color:transparent; background-clip:text; }
    .hero p { color:#b5c1dc; max-width:620px; margin:0 auto; line-height:1.7; font-size:.98rem; }
    [data-testid="stVerticalBlockBorderWrapper"] { border:1px solid #c9ddff24!important; border-radius:28px!important;
      background:linear-gradient(145deg,#ffffff12,#ffffff07 54%,#6b9aff0a)!important;
      box-shadow:0 28px 80px #0007, inset 0 1px 0 #ffffff12; backdrop-filter:blur(24px); padding:1.15rem; }
    .panel-heading { display:flex; justify-content:space-between; align-items:center; gap:1rem; margin-bottom:1.1rem; }
    .panel-heading h3 { margin:0; color:#f2f6ff; font:700 1.05rem Inter,'Segoe UI',sans-serif; }
    .panel-heading p { margin:.28rem 0 0; color:#91a1c4; font-size:.82rem; }
    .small-tag { color:#a9d7ff; font-size:.67rem; letter-spacing:.12em; border:1px solid #9ecbff2b; background:#66aaff10;
      border-radius:100px; padding:.38rem .65rem; white-space:nowrap; }
    [data-testid="stFileUploader"] { border:1px dashed #a6c6ff50; background:linear-gradient(150deg,#4d94ff12,#ffffff05);
      border-radius:20px; padding:1.1rem; }
    [data-testid="stFileUploader"] section { padding:.65rem .25rem; }
    [data-testid="stFileUploader"] small { color:#9eaccb!important; }
    [data-testid="stFileUploader"] button { border:1px solid #8dc5ff55!important; background:#2f75ec2b!important;
      color:#eaf5ff!important; border-radius:12px!important; font-weight:700!important; }
    .drop-copy { color:#b5c8eb; text-align:center; margin:.2rem 0 .8rem; font-size:.84rem; }
    .image-frame { border:1px solid #adcfff22; border-radius:19px; overflow:hidden; background:#060a16; min-height:250px; }
    [data-testid="stImage"] img { max-height:400px; object-fit:contain; border-radius:19px; }
    .empty-frame { height:280px; display:flex; align-items:center; justify-content:center; flex-direction:column;
      color:#8797bb; gap:.7rem; background:radial-gradient(ellipse at 50% 40%,#236ddd22,transparent 65%),#ffffff03; }
    .fabric-icon { width:66px; height:66px; display:grid; place-items:center; border-radius:22px; font-size:2rem;
      background:linear-gradient(145deg,#93dfff26,#3878ff21); border:1px solid #95c8ff2b; box-shadow:0 9px 30px #2879ff21; }
    .file-note { color:#8797bb; font-size:.74rem; margin-top:.65rem; }
    .result-card { border:1px solid #9ecbff27; border-radius:22px; padding:1.2rem 1.25rem; background:linear-gradient(145deg,#147cff27,#1d2c6b24 72%); }
    .result-label { color:#a8caef; font-size:.7rem; font-weight:700; letter-spacing:.16em; text-transform:uppercase; }
    .result-main { display:flex; align-items:center; justify-content:space-between; gap:1rem; margin-top:.55rem; }
    .result-name { color:#fff; font:800 clamp(1.6rem,3vw,2.15rem) Manrope,sans-serif; letter-spacing:-.04em; }
    .result-score { color:#c7f2ff; font:800 1.75rem Inter,'Segoe UI',sans-serif; }
    .result-caption { color:#9eafd0; font-size:.78rem; margin-top:.15rem; }
    .section-label { display:flex; justify-content:space-between; color:#e4ebfb; font-size:.79rem; font-weight:700; margin:1.35rem 0 .7rem; }
    .prob-row { display:grid; grid-template-columns:minmax(95px,1fr) 2fr 52px; align-items:center; gap:.7rem; margin:.65rem 0; }
    .prob-name { color:#c2cce3; font-size:.77rem; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .prob-track { height:7px; border-radius:99px; background:#ffffff12; overflow:hidden; }
    .prob-fill { height:100%; border-radius:99px; background:linear-gradient(90deg,#38bdf8,#4f8cff); box-shadow:0 0 12px #4f9aff60; }
    .prob-value { text-align:right; color:#d5e8ff; font:700 .75rem Inter,'Segoe UI',sans-serif; }
    .top-three { display:grid; grid-template-columns:repeat(3,1fr); gap:.55rem; margin-top:1rem; }
    .top-chip { padding:.65rem .7rem; border-radius:14px; border:1px solid #b5d1ff1c; background:#ffffff08; }
    .top-chip small { display:block; color:#8497bf; font-size:.65rem; margin-bottom:.2rem; }
    .top-chip strong { display:block; color:#eaf1ff; font-size:.77rem; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .top-chip span { color:#9fdcff; font:700 .75rem Inter,'Segoe UI',sans-serif; }
    .empty-result { min-height:375px; display:grid; place-items:center; text-align:center; border:1px solid #c9ddff17;
      border-radius:22px; background:radial-gradient(ellipse at 50% 35%,#267aff18,transparent 68%),#ffffff04; padding:1.5rem; }
    .empty-result h3 { color:#eef4ff; font:700 1.12rem Manrope,sans-serif; margin:.8rem 0 .35rem; }
    .empty-result p { max-width:300px; color:#94a4c6; font-size:.82rem; line-height:1.6; margin:0 auto; }
    .model-notice { border:1px solid #f4bf5d45; background:#f5b9420b; color:#f4d99b; border-radius:14px;
      padding:.8rem .95rem; margin-top:1rem; font-size:.8rem; line-height:1.55; }
    .footnote { color:#8190af; text-align:center; font-size:.72rem; margin:1.2rem auto 0; max-width:700px; line-height:1.6; }
    @media(max-width:700px) {
      .block-container{padding:1.2rem .9rem 2rem}
      .brandbar{margin-bottom:2rem}
      .online-pill{font-size:.68rem}
      .hero{margin-bottom:1.5rem}
      [data-testid="stHorizontalBlock"]{flex-direction:column;gap:1rem}
      [data-testid="stColumn"]{width:100%!important;flex:1 1 100%!important;min-width:100%!important}
      [data-testid="stVerticalBlockBorderWrapper"]{padding:.9rem}
      .prob-row{grid-template-columns:92px 1fr 48px}
      .top-chip{padding:.55rem .45rem}
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_predictor(checkpoint_mtime: float):
    del checkpoint_mtime
    return load_predictor(CHECKPOINT)


st.markdown(
    """
    <div class="brandbar">
      <div class="brand"><div class="brand-mark">✳</div><div>FABRIC AI<small>MATERIAL STUDIO</small></div></div>
      <div class="online-pill"><span class="online-dot"></span> RESNET18 · 7 MATERIALS</div>
    </div>
    <div class="hero">
      <div class="eyebrow">✦ &nbsp; AI FABRIC RECOGNITION</div>
      <h1>Khám phá chất liệu<br><span>trong từng thớ vải.</span></h1>
      <p>Tải ảnh bề mặt vải lên. Mô hình sẽ phân tích đặc trưng và hiển thị xác suất cho từng nhóm chất liệu.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([1, 1], gap="large")

with left:
    with st.container(border=True):
        st.markdown(
            '<div class="panel-heading"><div><h3>Ảnh đầu vào</h3><p>Chọn ảnh vải rõ nét để có kết quả tốt hơn</p></div>'
            '<div class="small-tag">01 &nbsp; UPLOAD</div></div><div class="drop-copy">Kéo thả ảnh vào đây hoặc chọn từ thiết bị</div>',
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader(
            "Chọn ảnh vải",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
            help="Hỗ trợ JPG, JPEG hoặc PNG, tối đa 12 MB.",
        )
        image = None
        image_error = None
        image_bytes = uploaded.getvalue() if uploaded is not None else None
        if uploaded is None:
            st.markdown(
                '<div class="image-frame empty-frame"><div class="fabric-icon">▦</div><div>Ảnh xem trước sẽ hiển thị tại đây</div></div>',
                unsafe_allow_html=True,
            )
            st.markdown('<div class="file-note">JPG / JPEG / PNG &nbsp; · &nbsp; Tối đa 12 MB</div>', unsafe_allow_html=True)
        elif len(image_bytes) > MAX_IMAGE_BYTES:
            image_error = "Ảnh vượt quá giới hạn 12 MB. Vui lòng chọn ảnh nhỏ hơn."
            st.error(image_error)
        else:
            try:
                image = Image.open(BytesIO(image_bytes)).convert("RGB")
                st.image(image, width="stretch")
                st.markdown(
                    f'<div class="file-note">{escape(uploaded.name)} &nbsp; · &nbsp; {image.width} × {image.height} px</div>',
                    unsafe_allow_html=True,
                )
            except (UnidentifiedImageError, OSError, ValueError):
                image_error = "Không thể đọc ảnh. Hãy chọn một file JPG, JPEG hoặc PNG hợp lệ."
                st.error(image_error)

with right:
    with st.container(border=True):
        st.markdown(
            '<div class="panel-heading"><div><h3>Kết quả nhận diện</h3><p>Phân bố xác suất theo từng chất liệu</p></div>'
            '<div class="small-tag">02 &nbsp; ANALYSIS</div></div>',
            unsafe_allow_html=True,
        )
        if image_error:
            st.markdown(
                '<div class="empty-result"><div><div class="fabric-icon">!</div><h3>Ảnh chưa sẵn sàng</h3><p>Chọn một ảnh hợp lệ để bắt đầu nhận diện.</p></div></div>',
                unsafe_allow_html=True,
            )
        elif image is None:
            st.markdown(
                '<div class="empty-result"><div><div class="fabric-icon">✧</div><h3>Sẵn sàng phân tích</h3><p>Khi bạn tải ảnh lên, kết quả dự đoán và độ tin cậy sẽ xuất hiện ở đây.</p></div></div>',
                unsafe_allow_html=True,
            )
            if not CHECKPOINT.exists():
                st.markdown(
                    '<div class="model-notice">Mô hình chưa được huấn luyện. Sau khi cài dependencies, chạy lệnh train trong README để bật nhận diện.</div>',
                    unsafe_allow_html=True,
                )
        elif not CHECKPOINT.exists():
            st.markdown(
                '<div class="empty-result"><div><div class="fabric-icon">⏳</div><h3>Ảnh đã sẵn sàng</h3><p>Chưa có checkpoint mô hình. Huấn luyện ResNet18 để bật kết quả nhận diện thật.</p></div></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="model-notice">Chạy <code>python -m src.train --epochs 20 --batch-size 32 --seed 42</code> trong thư mục project.</div>',
                unsafe_allow_html=True,
            )
        else:
            try:
                model, classes, transform, device = get_predictor(CHECKPOINT.stat().st_mtime)
                top, probabilities = predict_image(BytesIO(image_bytes), model, classes, transform, device, top_k=3)
                top_name, top_score = top[0]
                rows = []
                for name, score in sorted(zip(classes, probabilities), key=lambda item: item[1], reverse=True):
                    width = max(0, min(100, score * 100))
                    rows.append(
                        '<div class="prob-row"><div class="prob-name">{}</div><div class="prob-track">'
                        '<div class="prob-fill" style="width:{:.2f}%"></div></div><div class="prob-value">{:.1f}%</div></div>'.format(
                            escape(name), width, width
                        )
                    )
                chips = []
                for rank, (name, score) in enumerate(top, start=1):
                    chips.append(
                        '<div class="top-chip"><small>TOP {}</small><strong>{}</strong><span>{:.1f}%</span></div>'.format(
                            rank, escape(name), score * 100
                        )
                    )
                st.markdown(
                    '<div class="result-card"><div class="result-label">Most likely material</div>'
                    '<div class="result-main"><div><div class="result-name">{}</div>'
                    '<div class="result-caption">Kết quả có xác suất cao nhất</div></div>'
                    '<div class="result-score">{:.1f}%</div></div></div>'
                    '<div class="section-label"><span>Xác suất theo chất liệu</span><span>7 lớp</span></div>'
                    '{}<div class="top-three">{}</div>'.format(
                        escape(top_name), top_score * 100, "".join(rows), "".join(chips)
                    ),
                    unsafe_allow_html=True,
                )
            except Exception as exc:
                st.error(f"Không thể chạy mô hình: {exc}")

st.markdown(
    '<div class="footnote">Kết quả là xác suất dự đoán của mô hình trên ảnh đầu vào, không phải phép kiểm định thành phần sợi vải. '
    'Ánh sáng, góc chụp và chất lượng ảnh có thể ảnh hưởng đến dự đoán.</div>',
    unsafe_allow_html=True,
)

