import fitz  # PyMuPDF

def extract_pages(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    pages = []
    for i in range(doc.page_count):
        text = doc[i].get_text()
        pages.append(text)
    return pages
