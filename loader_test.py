import os
from namu_crawler import NamuCrawler
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
from typing import Iterator
import time
base_url = "https://namu.wiki"


url = 'https://namu.wiki/w/ILLIT'
max_hop = 1
verbose = True
base_nc = NamuCrawler(url = url, hop = 0)
base_nc.construct_toc()
if (verbose) == True:
    base_nc.print_toc()

def get_total_content(parent_item, sub_url, hop):
    """sub crawler 생성 및 데이터 가져오기"""
    sub_nc = NamuCrawler(url = sub_url, hop = hop)
    sub_nc.construct_toc()
    # print(sub_nc.get_doc_title(), parent_item, sub_nc.hop, max_hop)
    to_return = ""
    start = time.time()
    for k, v in sub_nc.toc_dict.items():
        cur_toc_item, content = sub_nc.get_content_heading(k)
        
        if type(content) == str and f'/w/' in content: # content가 링크 대체이면서 
            if sub_nc.hop < max_hop: #현재 문서의 hop이 max_hop보다 적거나 같으면 더 들어가기
                content = get_total_content(parent_item = parent_item, sub_url = base_url + content, hop = sub_nc.hop + 1)
            else: # max_hop과 같으면 그냥 링크로 대체한다고만 써주기
                content = f"{cur_toc_item[1]}: 다음 문서로 대체 설명: {base_url + content}"
        elif content == None: # 빈 각주라면 그냥 넘어가기
            continue
        else: # 일반 설명은 {현재 목차 : 설명} 꼴로 구성
            content = f'{cur_toc_item[1]}: {" ".join(content) if type(content) == list else content}'
        
        to_return = to_return + "\n" + content + "\n"

    # if verbose:
    print(f"Sub document {sub_nc.get_doc_title()} done (hop : {sub_nc.hop}/{max_hop} \t elapsed_time: {round(time.time() - start, 1)} seconds)")
    return to_return

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
        parent_toc_items = ""
    elif parent_meta['parent_toc_items'] == "":
        parent_toc_items = parent_meta['toc_item']
    else:
        parent_toc_items = parent_meta['parent_toc_items'] + "/" +  parent_meta['toc_item']
    meta_data = {
        "base_url" : parent_meta['base_url'] if parent_meta else nc.url,
        "parent_url" : parent_meta['cur_url'] if parent_meta else None,
        "cur_url" : nc.url,
        "page_hop" : nc.hop,
        "parent_index" : parent_meta['index'] if parent_meta else None,
        "parent_toc_items" : parent_toc_items,
        "index" : cur_toc_item[0],
        "toc_item" : cur_toc_item[1],
    }

    if content == None:
        return [Document(page_content = None, metadata = meta_data)]

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