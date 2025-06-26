import streamlit as st
import base64

def auto_download(markdown_text: str, file_name: str):
    """
    마크다운 텍스트를 base64로 인코딩하여 브라우저에서 자동 다운로드 트리거를 발생시킵니다.
    Streamlit에서 사용하는 HTML + JavaScript 기반 다운로드 방식입니다.
    """
    b64 = base64.b64encode(markdown_text.encode()).decode()
    download_html = f"""
    <html>
      <body>
        <a id="downloadLink" href="data:text/markdown;base64,{b64}" download="{file_name}" style="display:none">Download</a>
        <script>document.getElementById('downloadLink').click();</script>
      </body>
    </html>
    """
    st.components.v1.html(download_html, height=0)
