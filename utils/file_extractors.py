import os
import io
import zipfile
import streamlit as st
import pandas as pd
import xml.etree.ElementTree as ET
import pdfplumber
from pdfminer.layout import LAParams


def extract_text_from_pdf(file):
    file_bytes = file.read()
    pdf_io = io.BytesIO(file_bytes)
    text = ""
    laparams = LAParams(char_margin=2.0, line_margin=0.5, word_margin=0.1, boxes_flow=0.5)
    with pdfplumber.open(pdf_io, laparams=laparams.__dict__) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

            tables = page.extract_tables()
            if tables:
                for table in tables:
                    table = fill_missing_cells(table)
                    try:
                        df = pd.DataFrame(table[1:], columns=table[0])
                        markdown_table = df.to_markdown(index=False)
                    except Exception:
                        markdown_table = "\n".join([" | ".join(map(str, row)) for row in table])
                    text += "\n" + markdown_table + "\n\n"
    return text


def fill_missing_cells(table):
    if not table or len(table) < 2:
        return table
    for i in range(1, len(table)):
        row = table[i]
        for j in range(len(row)):
            if row[j] in [None, ""]:
                row[j] = table[i - 1][j] if table[i - 1][j] not in [None, ""] else ""
    return table


def extract_text_from_hwpx(file):
    try:
        file_bytes = file.read()
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
            names = z.namelist()
            candidate = next((name for name in names if "section0.xml" in name.lower()), None)
            if not candidate:
                st.error("hwpx 파일에서 section XML을 찾을 수 없습니다.")
                return ""
            xml_content = z.read(candidate)
    except Exception as e:
        st.error(f"hwpx 파일 처리 오류: {e}")
        return ""
    try:
        tree = ET.fromstring(xml_content)
    except Exception as e:
        st.error(f"XML 파싱 오류: {e}")
        return ""
    texts = [elem.text.strip() for elem in tree.iter() if elem.text]
    return " ".join(texts).strip()


def extract_text_from_txt(file):
    try:
        content = file.read()
        if isinstance(content, bytes):
            content = content.decode("utf-8")
        return content
    except Exception as e:
        st.error(f"TXT 파일 추출 오류: {e}")
        return ""


def extract_text_from_pptx(file):
    try:
        from pptx import Presentation
    except ImportError:
        st.error("python-pptx가 설치되어 있지 않습니다. pip install python-pptx")
        return ""
    try:
        file.seek(0)
        ppt = Presentation(io.BytesIO(file.read()))
        text = ""
        for slide in ppt.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text:
                    text += shape.text + "\n"
        return text
    except Exception as e:
        st.error(f"PPTX 파일 추출 오류: {e}")
        return ""


def extract_text_from_xlsx(file):
    try:
        from openpyxl import load_workbook
        file_bytes = file.read()
        wb = load_workbook(filename=io.BytesIO(file_bytes), data_only=True)
        result_text = ""
        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            merged_ranges = list(ws.merged_cells.ranges)
            for merged_range in merged_ranges:
                top_left_value = ws.cell(merged_range.min_row, merged_range.min_col).value
                ws.unmerge_cells(range_string=str(merged_range))
                for row in range(merged_range.min_row, merged_range.max_row + 1):
                    for col in range(merged_range.min_col, merged_range.max_col + 1):
                        ws.cell(row=row, column=col).value = top_left_value
            data = ws.values
            columns = next(data)
            df = pd.DataFrame(data, columns=columns)
            try:
                markdown_table = df.to_markdown(index=False)
            except Exception:
                markdown_table = df.to_string(index=False)
            result_text += f"Sheet: {sheet_name}\n{markdown_table}\n\n"
        return result_text
    except Exception as e:
        st.error(f"XLSX 파일 추출 오류: {e}")
        return ""


def extract_text_from_file(file):
    ext = os.path.splitext(file.name)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file)
    elif ext == ".hwp":
        st.error("HWP 파일 추출 기능은 지원되지 않습니다.")
        return ""
    elif ext == ".hwpx":
        return extract_text_from_hwpx(file)
    elif ext == ".txt":
        return extract_text_from_txt(file)
    elif ext == ".pptx":
        return extract_text_from_pptx(file)
    elif ext == ".xlsx":
        return extract_text_from_xlsx(file)
    else:
        st.error("지원되지 않는 파일 형식입니다. (PDF, HWPX, TXT, PPTX, XLSX)")
        return ""
