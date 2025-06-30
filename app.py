import streamlit as st
from utils.web_crawler import handle_web_crawl_tab
from utils.file_extractors import handle_file_to_md_tab
from utils.markdown_converter import handle_md_to_txt_tab

def main():
    st.set_page_config(page_title="통합 변환기", layout="wide")
    st.title("통합 변환기")
    st.write("웹 크롤링, 파일 변환, Markdown → TXT 변환 기능을 제공합니다.")

    tabs = st.tabs(["Web Crawl to Markdown", "File to Markdown", "Markdown to TXT 변환"])

    with tabs[0]:
        handle_web_crawl_tab()

    with tabs[1]:
        handle_file_to_md_tab()

    with tabs[2]:
        handle_md_to_txt_tab()

if __name__ == "__main__":
    main()
