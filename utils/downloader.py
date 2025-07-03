import streamlit as st
import base64
import re

def sanitize_filename(file_name: str) -> str:
    """파일 이름에서 안전하지 않은 문자 제거"""
    return re.sub(r'[^a-zA-Z0-9_\-\.]', '_', file_name)

def auto_download(markdown_text: str, file_name: str):
    """
    마크다운 텍스트를 base64로 인코딩하여 사용자가 클릭 시 다운로드할 수 있게 합니다.
    보안과 UX를 고려하여 자동 클릭 대신 명시적인 버튼 사용.
    """
    file_name = sanitize_filename(file_name)
    b64 = base64.b64encode(markdown_text.encode('utf-8')).decode()
    href = f'<a href="data:text/markdown;charset=utf-8;base64,{b64}" download="{file_name}">📥 파일 다운로드 ({file_name})</a>'
    st.markdown(href, unsafe_allow_html=True)