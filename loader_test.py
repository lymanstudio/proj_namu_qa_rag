import os
from namu_crawler import NamuCrawler
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
from typing import Iterator
import time
base_url = "https://namu.wiki"


url = 'https://namu.wiki/w/프로미스 나인'
max_hop = 2
verbose = True
base_nc = NamuCrawler(url = url, hop = 0)
base_nc.construct_toc()
if (verbose) == True:
    base_nc.print_toc()


def make_sub_nc(parent_nc, sub_url, hop, parent_meta):
    sub_nc = NamuCrawler(url = sub_url, hop = hop)
    sub_nc.construct_toc()
    to_return_docs = []
    for s, header in sub_nc.toc_dict.items():
        to_return_docs += get_docs(sub_nc, s, parent_meta)

    return to_return_docs

def get_docs(nc, s, parent_meta = None):
    """본문 내용 가져오기"""
    cur_toc_item, content = nc.get_content_heading(s)
    if parent_meta == None:
        parent_page_index = ""
        parent_page_toc_item = ""
        abs_page_toc_items = (nc.get_ancestor_items(toc_index = cur_toc_item[0]) + "/" ) + cur_toc_item[1] if nc.get_ancestor_items(toc_index = cur_toc_item[0]) != '' else '' + cur_toc_item[1]
        
    else: #  parent_meta['parent_page_toc_item'] == "":
        parent_page_index = parent_meta['index']
        parent_page_toc_item = parent_meta['toc_item']
        abs_page_toc_items = (
            parent_meta['abs_page_toc_item'] 
            + "//" 
            + ((nc.get_ancestor_items(toc_index = cur_toc_item[0]) + "/" ) if nc.get_ancestor_items(toc_index = cur_toc_item[0]) != '' else '')
            + cur_toc_item[1]
        )
    # else:
    #     parent_page_index = parent_meta['parent_page_index'] + "/" +  parent_meta['index']
    #     parent_page_toc_item = parent_meta['parent_page_toc_item'] + "/" +  parent_meta['toc_item']
        

    meta_data = {
        "base_page_url" : parent_meta['base_page_url'] if parent_meta else nc.url,
        "parent_page_url" : parent_meta['current_page_url'] if parent_meta else None,
        "current_page_url" : nc.url,
        "page_hop" : nc.hop,

        "parent_page_index" : parent_page_index,
        "parent_page_toc_item" : parent_page_toc_item,
        "abs_page_toc_item": abs_page_toc_items,

        "index" : cur_toc_item[0],
        "toc_item" : cur_toc_item[1],
        "ancestor_toc_items" : nc.get_ancestor_items(toc_index = cur_toc_item[0])
    }

    if content == None:
        return [Document(page_content = "", metadata = meta_data)]

    elif type(content) == str and '/w/' in content: 
        # content가 링크 대체라면 
        if nc.hop + 1 <= max_hop: # max_hop보다 아래라면 링크로 들어가 전체 다 가져오기
            print(base_url + content)
            sub_docs = make_sub_nc(
                parent_nc = nc, 
                sub_url = base_url + content, 
                hop = nc.hop + 1,
                parent_meta = meta_data
            )
            return sub_docs
        else: # 아니라면 설명으로 대체
            return  [Document(page_content = f"다음 문서로 대체 설명: {base_url + content}", metadata = meta_data)]
    else:
        return [Document(page_content = " ".join(content), metadata = meta_data)]


docs = []


for s, header in base_nc.toc_dict.items():
    cur_docs = get_docs(base_nc, s)
    docs += cur_docs
    
    # if cur_content == None: #각주가 없는 문서의 경우 cur_content를 None 반환 => 넘어가기
    #     continue

    # if verbose == True:
    #     print(">> ", s, header[0][1])
    #     print("\t", cur_content)
    # docs.append(Document(page_content= cur_content, metadata = cur_metadata))

def print_doc(docs):
    
    for doc in docs:
        if len(doc.page_content) > 200:
            page_content = doc.page_content[:200]
        else:
            page_content = doc.page_content
        print(f"metadata: {doc.metadata}")
        print(f"content: {page_content}")
        print("==================================================")

def export_csv(docs):
    import pandas as pd

    tmp_df = pd.DataFrame(columns = list(docs[0].metadata.keys()))
    for doc in docs:
        if len(doc.page_content) > 200:
            page_content = doc.page_content[:200]
        else:
            page_content = doc.page_content
        cur_data = doc.metadata
        cur_data['page_content']= page_content
        tmp_df = pd.concat([tmp_df, pd.DataFrame([cur_data])], ignore_index=True)
    return tmp_df


    

from namu_loader import NamuLoader

url = 'https://namu.wiki/w/ILLIT'
nl = NamuLoader(url = url, max_hop = 2, verbose = True)
docs2= nl.load()

df = export_csv(docs2)
df.to_csv("tmp_df.csv")
