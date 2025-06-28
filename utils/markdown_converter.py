import re
import os
from openai import OpenAI
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def convert_chunk_to_markdown(current_chunk, previous_md=None):
    """
    현재 청크 텍스트를 Markdown 문서로 변환합니다.
    이전 청크의 문맥 일부를 참고하여 자연스러운 연결을 유도할 수 있습니다.
    """
    context = ""
    if previous_md:
        prev_words = previous_md.split()
        context = " ".join(prev_words[-100:]) + "\n\n"

    prompt = (
        "**현재청크**를 잘 구조화된 Markdown 문서로 변환해줘. "
        "Markdown 문서의 문맥을 분석하여 어색한 부분들은 모두 삭제해줘. "
        "**이전 마크다운 청크**는 포함하지 않고 **현재청크**만 정제해서 만들어줘. "
        f"**이전 마크다운 청크**: {context}\n\n"
        f"**현재청크**: {current_chunk}"
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "너는 텍스트 변환 전문가야. 주어진 청크의 모든 세부 정보를 보존하면서 Markdown 문서로 변환해."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    markdown_text = response.choices[0].message.content
    return remove_code_fence(markdown_text)


def transform_level1_header(md_content):
    """
    Markdown 텍스트에서 레벨 1 헤더(`#`)를 '@!@' 기호로 치환해 일반 텍스트 형태로 변환합니다.
    """
    return re.sub(r'^( *)#(?!#)\s+', r'\1@!@ ', md_content, flags=re.MULTILINE)


def remove_code_fence(markdown_text):
    """
    GPT 응답에서 불필요한 ``` 코드 펜스를 제거합니다.
    """
    markdown_text = markdown_text.strip()
    if markdown_text.startswith("```"):
        lines = markdown_text.splitlines()
        if lines[0].startswith("```"):
            if lines[-1].strip().startswith("```"):
                markdown_text = "\n".join(lines[1:-1]).strip()
            else:
                markdown_text = "\n".join(lines[1:]).strip()
    return markdown_text


def remove_isolated_code_fences(text):
    """
    단독으로 존재하는 ``` 또는 ```markdown 태그를 제거합니다.
    """
    return re.sub(r"(?m)^\s*```(markdown)?\s*$", "", text)

def handle_md_to_txt_tab():
    st.header("Markdown to TXT 변환기")
    uploaded_file = st.file_uploader("Markdown 파일 업로드", type=["md"])
    if uploaded_file is not None:
        md_text = uploaded_file.read().decode("utf-8")
        st.text_area("미리보기", md_text, height=300)
        if st.button("TXT로 변환"):
            st.download_button("다운로드", data=md_text, file_name="converted.txt")
