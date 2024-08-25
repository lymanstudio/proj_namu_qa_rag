import requests
from bs4 import BeautifulSoup
import re
import urllib.parse


# 웹 페이지 URL
url = 'https://namu.wiki/w/%EC%84%9C%EC%9A%B8%20%EB%B2%84%EC%8A%A4%205413'
# url = 'https://namu.wiki/w/ILLIT'
# url = 'https://namu.wiki/w/SUMMER%20MOON'
# url = 'https://namu.wiki/w/%EA%B4%91%EC%A0%80%EC%9A%B0%20%EC%B0%A8%EC%A7%80'

# url = 'https://namu.wiki/w/ILLIT/Weverse'

# html 가져오기
session = requests.Session()
html_doc = session.get(url)
soup = BeautifulSoup(html_doc.text, "html.parser")

def get_keywords_with_neighbors(text, keyword, hop = 10):
    pattern = re.compile(keyword)
    matches = pattern.finditer(text)

    for match in matches:
        print("▶︎\t", text[match.start() - hop: match.end() + hop])

def get_class_name(keyword, soup):
    target_elements = soup.find_all(text=keyword)

    # 찾은 태그들의 클래스 이름 출력
    for element in target_elements:
        parent = element.parent  # 텍스트의 부모 태그를 가져옴
        if 'class' in parent.attrs:
            print(parent['class'])
        else:
            print('No class attribute')


"""## 목차(TOC) 정보 추출 및 TOC 딕셔너리 구성"""
toc = soup.find("div", class_ = 'toc-indent')
toc_dict = dict()

"""만약 프로필이 있는 문서라면 가장 앞에 구성"""
html_str = str(soup)
start_pos = html_str.find(str('<body>')) + len(str('<body>'))
toc_str = str(soup.find("div", class_='wiki-macro-toc'))
end_pos = html_str.find(toc_str)
between_content = html_str[start_pos:end_pos]
soup_between = BeautifulSoup(between_content, 'html.parser')

if len(profile := soup_between.find_all('div', class_ = 'wiki-table-wrap table-right')) > 0:
    toc_dict['s-p'] = (("0.",'PROFILE'), profile[-1]) # dictionary의 첫번째 아이템으로 넣기


"""목차 아이템들을 하나씩 딕셔너리에 저장(key = s-#.#.#, value = (목차 명, 목차 element))"""
pattern = r'^(\d+\.)+' # 제목 파싱용 패턴

for i, e in enumerate(toc.find_all("span", class_  = "toc-item")):
    item_value = e.get_text()
    numbers = re.match(pattern, item_value).group()
    text = re.sub(pattern, '', item_value).strip()
    toc_dict[e.find('a')['href'].replace("#", "")] = ((numbers, text), soup.find('a', id=e.find('a')['href'].replace("#", "")))

"""마지막 원소로 각주영역 저장, 없다면 None 저장"""
toc_dict['s-f'] = (("EOD.", 'FOOTNOTES'), soup.find("div", class_ = 'wiki-macro-footnote')) # 마지막엔 각주 영역

# 목차 프린트
print(f"\nDocument of {soup.find("a", href = "/w/"+url.split("/w/")[-1].split("?")[0]).get_text()}\n")
print("\n================== Table of Contents  ==================\n")
for k, v in toc_dict.items():
    num = v[0][0].count(".")
    if num == 0:
        print(v[0][0] + " " + v[0][1])
    else:
        print("".join(["\t"] * (num - 1)) + v[0][0] + " " + v[0][1])
def get_doc_title(url) -> str:
    """URL에서 현재 문서의 타이틀(주제) 반환 """
    topic = url.split("/w/")[-1].split("?")[0]
    topic = urllib.parse.unquote(topic)
    return soup.find("a", href = "/w/"+topic).get_text()
def get_ancestor_items(toc_index, level = None):
    toc_index = toc_index[:-1] if toc_index [-1] == "." else toc_index
    
    if level == None:
        level = len(toc_index.split(".")) - 1
        
    cur_toc_index = toc_index
    ancestors = []
    for i in range(level):
        ancestors.insert(0, toc_dict.get(f"s-{".".join(cur_toc_index.split(sep = ".")[:-1])}")[0])
        cur_toc_index = ancestors[-1][0][:-1]
    toc_items = "/".join([i[1] for i in ancestors])
    return toc_items

def strip_footnotes(ele):
    content_list = []
    prv_tag = None
    for c in ele.descendants:
        if c.name == 'a' and 'wiki-fn-content' in c.get('class', []):
            # 'wiki-fn-content' 클래스를 가진 'a' 태그를 리스트에 추가
            content_list.append(f"({c['href']}; {c['title']})")
            prv_tag = True
        elif isinstance(c, str):
            # 텍스트 노드를 리스트에 추가
            if prv_tag == True:
                prv_tag = False
                continue
            content_list.append(c.string.strip())
            prv_tag = False
    return content_list

def get_content_between_tags(head, start_tag, end_tag):
    """두개의 태그 사이의 wiki-paragraph 정보 추출"""
    html_str = str(soup)


    # 시작 태그와 끝 태그의 위치를 찾아 사이의 컨텐츠를 추출, 임시 soup로 만듦
    start_pos = html_str.find(str(start_tag))
    end_pos = html_str.find(str(end_tag))
    between_content = html_str[start_pos:end_pos]
    soup_between = BeautifulSoup(between_content, 'html.parser')

    # wiki-paragraph를 가진 엘리먼트를 수집, 텍스트 컨텐츠 추출
    elements_between = soup_between.find_all('div', class_='wiki-paragraph')

    if len(elements_between) == 0:
        # 설명이 아예 없는 경우 None 반환
        return (head, None)
    elif (ext_icon := elements_between[0].find("img", alt = '상세 내용 아이콘')) != None:
        # 타 문서로 설명이 대체된 경우엔 링크 반환
        ext_link = elements_between[0].find("a", class_ = "wiki-link-internal")['href']
        return (head, ext_link)
    else: 
        # 설명이 있는 경우엔 get_text()로 텍스트 반환
        text_content = []
        for element in elements_between:
            # 만약 각주가 있는 엘리먼트라면 각주를 strip하는 함수 적용,  아니면 그냥 일반 get_text() 적용
            if element.find("a", class_ = 'wiki-fn-content') != None:
                text_content.append(strip_footnotes(element))
            else:
                text_content.append(element.get_text())
            
        return (head, text_content)
    
def get_item_and_next(target_key):
    """toc_dict에서 현재 헤드의 다음 헤드 값을 반환"""
    keys = list(toc_dict.keys())
    values = list(toc_dict.values())
    
    if target_key in toc_dict:
        index = keys.index(target_key)
        if index < len(keys) - 1:
            return keys[index + 1], values[index + 1]
        else:
            return {keys[index]: values[index]}  # 마지막 키인 경우 다음 아이템이 없음
    else:
        return None

def get_content_heading(heading_idx):
    """헤드 값의 컨텐츠 내용을 반환"""
    start_tag = toc_dict.get(heading_idx)[1]
    try:
        end_tag = get_item_and_next(heading_idx)[1][1] 
    except(KeyError): #마지막 헤딩(보통 각주영역)의 경우엔 직접 로딩
        if toc_dict.get(heading_idx)[1] == None: # 각주가 없는 문서
            return (toc_dict.get(heading_idx)[0], None)
        else:
            return (toc_dict.get(heading_idx)[0], toc_dict.get(heading_idx)[1].get_text())
    content = get_content_between_tags(head = toc_dict.get(heading_idx)[0], start_tag= start_tag, end_tag= end_tag)

    return content



html_str = str(soup)


# 시작 태그와 끝 태그의 위치를 찾아 사이의 컨텐츠를 추출, 임시 soup로 만듦
start_pos = html_str.find(str(toc_dict.get("s-4")[1]))
end_pos = html_str.find(str(toc_dict.get("s-5")[1]))
between_content = html_str[start_pos:end_pos]
soup_between = BeautifulSoup(between_content, 'html.parser')
elements_between = soup_between.find_all('div', class_='wiki-paragraph')

get_content_heading("s-4")


