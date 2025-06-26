import streamlit as st
import time

from utils.common import process_url
from utils.downloader import auto_download


def handle_web_crawl_tab():
    st.header("웹 크롤링 → Markdown 변환")
    st.write(
        "여러 URL을 입력하고 (각 줄에 하나씩) 옵션으로 특정 클래스명을 지정하면, "
        "각 URL의 동적 페이지를 렌더링하여 텍스트와 <a>, <img> 등의 HTML 마크업을 그대로 추출한 후, "
        "OpenAI를 통해 Markdown으로 정제한 결과를 다운로드합니다."
    )

    urls_text = st.text_area("크롤링할 URL들을 입력하세요 (각 줄에 하나씩):", "https://example.com")
    target_class = st.text_input("특정 클래스명을 입력하세요 (옵션):", "")

    if st.button("문서 생성 (Web Crawl)"):
        urls = [line.strip() for line in urls_text.splitlines() if line.strip()]
        if not urls:
            st.error("적어도 하나의 URL을 입력하세요.")
        else:
            for url in urls:
                st.info(f"처리 중: {url}")
                try:
                    file_name, md_text = process_url(url, target_class.strip() or None)
                    if not md_text:
                        st.warning(f"{url}에서 콘텐츠를 추출하지 못했습니다.")
                        continue
                    st.success(f"{url} → 생성된 파일: {file_name}")
                    auto_download(md_text, file_name)
                    st.markdown("---")
                    time.sleep(1)
                except Exception as e:
                    st.error(f"{url} 처리 중 오류 발생: {e}")
