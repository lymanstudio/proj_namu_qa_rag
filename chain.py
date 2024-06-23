import os
base_dir = "c:/Users/thdwo/Documents/Github/proj_artist_info_gen"
os.chdir("../")
print(os.getcwd())
from dotenv import load_dotenv
load_dotenv(dotenv_path= os.path.join(os.getcwd(), "keys/.env"))
os.chdir(base_dir)

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import bs4
loader = WebBaseLoader(
    web_path= "https://namu.wiki/w/%ED%94%84%EB%A1%9C%EB%AF%B8%EC%8A%A4%EB%82%98%EC%9D%B8",
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("toc-item")
        )
    ),
)

def get_document_from_web(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
        )
    splitDocs = splitter.split_documents(docs)
    return splitDocs


docs = get_documnet_from_web("https://namu.wiki/w/%ED%94%84%EB%A1%9C%EB%AF%B8%EC%8A%A4%EB%82%98%EC%9D%B8")



import re

def get_keywords_with_neighbors(text, keyword, hop = 10):
    pattern = re.compile(keyword)
    matches = pattern.finditer(text)

    for match in matches:
        print("▶︎\t", text[match.start() - hop: match.end() + hop])
