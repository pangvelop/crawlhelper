import re
import datetime
from urllib.parse import urljoin

from bs4 import BeautifulSoup, NavigableString
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

from openai import OpenAI
import os

# OpenAI 클라이언트 설정
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def split_text_with_overlap(text, chunk_size, overlap):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start = end - overlap
    return chunks


def remove_code_block_markers(text: str) -> str:
    text = re.sub(r'^```(?:\w+)?\n', '', text)
    text = re.sub(r'\n```$', '', text)
    return text


def remove_code_fence(markdown_text):
    markdown_text = markdown_text.strip()
    if markdown_text.startswith("```"):
        lines = markdown_text.splitlines()
        if lines[-1].strip().startswith("```"):
            markdown_text = "\n".join(lines[1:-1]).strip()
        else:
            markdown_text = "\n".join(lines[1:]).strip()
    return markdown_text


def remove_isolated_code_fences(text):
    return re.sub(r"(?m)^\s*```(markdown)?\s*$", "", text)


###############################################
# 웹 크롤링 관련 공통 함수
###############################################
def get_rendered_html(url):
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(url)
    import time; time.sleep(3)
    html = driver.page_source
    driver.quit()
    return html


def extract_element(element, base_url=None):
    if isinstance(element, NavigableString):
        return element.strip()
    if element.name in ['script', 'style']:
        return ""
    if element.name in ['a', 'img']:
        if element.name == 'img' and base_url:
            src = element.get('src')
            if src and not src.startswith('http'):
                element['src'] = urljoin(base_url, src)
        return str(element)
    result = ""
    for child in element.children:
        child_text = extract_element(child, base_url)
        if child_text:
            result += child_text + " "
    return result.strip()


def extract_content(html, target_class=None, base_url=None):
    soup = BeautifulSoup(html, 'html.parser')
    container = soup.find(class_=target_class) if target_class else soup
    return extract_element(container, base_url)


def convert_to_markdown(text):
    prompt = (
            "다음 텍스트에서 문맥상 필요없는 부분이나 읽기 어려운 부분, 그리고 javascript 등 코딩 코드를 제거하고 "
            "적절한 헤딩, 단락, 리스트 등을 사용하여 잘 구조화된 Markdown 문서를 만들어줘. "
            "단, 마지막에 추가적인 요약이나 정리 멘트는 생략해줘:\n\n" + text
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "너는 평문을 Markdown 형식으로 정제하는 전문가야. 응답은 마크다운 형식 그대로여야 해."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content


def process_url(url, target_class):
    html = get_rendered_html(url)
    content = extract_content(html, target_class, base_url=url)
    if not content:
        return None, None
    md_text = convert_to_markdown(content)
    md_text = remove_code_block_markers(md_text)
    header_line = next((line.strip() for line in md_text.splitlines() if line.strip().startswith("#")), None)
    sanitized_header = re.sub(r'[\\/*?:"<>|]', "", header_line.lstrip("#").strip()) if header_line else "converted_document"
    now = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    file_name = f"{sanitized_header}_{now}.md"
    return file_name, md_text
