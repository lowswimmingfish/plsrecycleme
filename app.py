import streamlit as st
import pandas as pd
from PIL import Image
import pytesseract
import cv2
import numpy as np
import openai
import io
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib

# ✅ 한글 폰트 설정
matplotlib.rcParams['font.family'] = 'AppleGothic'
matplotlib.rcParams['axes.unicode_minus'] = False

# ✅ OpenAI API 키 불러오기
openai.api_key = st.secrets["openai"]["api_key"]

# ✅ GPT 프롬프트 (간편장부 기준)
def ask_gpt_for_journal_entries(ocr_text):
    system_prompt = (
        "너는 소규모 동아리/모임의 회계 담당자야. 이 모임은 영리 활동을 하지 않고, "
        "회원 회비나 간단한 외부 지원금으로 운영돼. 따라서 기업 회계에서 쓰는 '매출', '수익', '비용' 같은 용어는 쓰면 안 돼.\n\n"
        "지출은 행사비, 식비, 교통비, 소모품비, 물품구입비 등으로 구분하고, "
        "입금은 회비수입, 지원금, 개인선납금 등으로 분류해줘.\n\n"
        "결과는 표로 정리해줘 이게 매우 중요해. 항목명 / 계정과목 / 차변 또는 대변 / 금액 형식으로.\n\n"
        "예시:\n"
        "- 편의점 간식 구입 → 항목명: 간식, 계정과목: 식비, 차변, 7,000원\n"
        "- 회비 입금 → 항목명: 회비입금, 계정과목: 회비수입, 대변, 10,000원\n"
        "- 카드결제 → 계정과목: 보통예금 또는 카드, 대변\n"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": ocr_text}
    ]

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ GPT 요청 실패: {e}"

# ✅ GPT 마크다운 표 파싱 함수
def parse_markdown_table(markdown_text):
    try:
        lines = markdown_text.strip().split("\n")
        table_lines = [line for line in lines if "|" in line]
        if len(table_lines) < 2:
            return pd.DataFrame()

        table = "\n".join(table_lines)
        df = pd.read_csv(io.StringIO(table), sep="|", engine="python", skipinitialspace=True)
        df = df.dropna(axis=1, how="all")
        df.columns = [col.strip() for col in df.columns]
        df = df.drop(df.index[0])
        df = df.reset_index(drop=True)
        return df
    except Exception as e:
        st.warning(f"⚠️ 표 변환 실패: {e}")
        return pd.DataFrame()

# ✅ Streamlit UI 시작
st.title("📸 자동분개 시스템")

uploaded_files = st.file_uploader("영수증 이미지 업로드", accept_multiple_files=True, type=["png", "jpg", "jpeg"])

all_dataframes = []

if uploaded_files:
    for file in uploaded_files:
        image = Image.open(file)
        st.image(image, caption="업로드한 이미지", use_container_width=True)

        # ✅ 이미지 전처리
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        img = cv2.medianBlur(img, 3)
        scale_percent = 150
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
        thresh = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        # ✅ OCR 수행
        text = pytesseract.image_to_string(thresh, lang="kor+eng")
        ocr_input = st.text_area("📜 OCR 결과 (수정 가능)", text, height=200, key=file.name)

        # ✅ GPT 재요청 버튼
        if st.button(f"🤖 GPT 재요청 ({file.name})"):
            gpt_result = ask_gpt_for_journal_entries(ocr_input)
            st.session_state[f"gpt_result_{file.name}"] = gpt_result

        gpt_result = st.session_state.get(f"gpt_result_{file.name}", ask_gpt_for_journal_entries(ocr_input))

        st.subheader("🤖 GPT 자동분개 결과")
        st.text_area("GPT 응답", gpt_result, height=300, key=f"gpt_out_{file.name}")

        # ✅ GPT 응답 → 표 변환
        df = parse_markdown_table(gpt_result)
        if not df.empty:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            df["수정시간"] = now
            st.subheader("📊 자동분개 표 (수정 가능)")
            edited_df = st.data_editor(df, num_rows="dynamic", key=f"editor_{file.name}")

            # ✅ CSV 다운로드
            csv = edited_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="💾 CSV로 저장하기",
                data=csv,
                file_name=f"auto_journal_{file.name}.csv",
                mime="text/csv",
            )

            all_dataframes.append(edited_df)

if all_dataframes:
    df_concat = pd.concat(all_dataframes, ignore_index=True)
    try:
        df_concat["금액"] = df_concat["금액"].astype(str).str.replace(",", "").str.replace("원", "").str.strip()
        df_concat["금액"] = pd.to_numeric(df_concat["금액"], errors="coerce")

        # 유연한 열 이름 인식
        debit_col = None
        for col in df_concat.columns:
            if "차변" in col and "대변" in col:
                debit_col = col
                break

        if debit_col is None:
            raise ValueError("‘차변 또는 대변’ 관련 열을 찾을 수 없음")

        debit_df = df_concat[df_concat[debit_col].str.contains("차변", na=False)]
        grouped = debit_df.groupby("계정과목")["금액"].sum()

        fig, ax = plt.subplots(figsize=(6, 6))
        wedges, texts, autotexts = ax.pie(grouped, labels=grouped.index, autopct="%1.1f%%", startangle=90, textprops=dict(color="black"))
        ax.axis("equal")
        st.subheader("🥧 계정과목별 지출 비율")
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Pie chart 시각화 실패: {e}")
