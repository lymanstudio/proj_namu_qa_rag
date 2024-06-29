import requests
from bs4 import BeautifulSoup
import re

# 웹 페이지 URL
# url = 'https://namu.wiki/w/%EC%9A%A9%EC%9D%B8%20%EB%B2%84%EC%8A%A4%20119'
# url = 'https://namu.wiki/w/ILLIT'
url = 'https://namu.wiki/w/%EC%96%B4%EB%A6%B0%EC%9D%B4%EB%B3%B4%ED%98%B8%EA%B5%AC%EC%97%AD'

url = 'https://namu.wiki/w/Pok%C3%A9Rogue?from=%ED%8F%AC%EC%BC%80%EB%A1%9C%EA%B7%B8'
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


## 목차(TOC) 정보 추출 및 TOC 딕셔너리 구성
toc = soup.find_all("div", class_ = 'toc-indent')[0]
toc_dict = dict()

# 만약 프로필이 있는 문서라면 가장 앞에 구성
html_str = str(soup)
start_pos = html_str.find(str('<body>')) + len(str('<body>'))
end_pos = html_str.find(str('<div class="wiki-macro-toc"'))
between_content = html_str[start_pos:end_pos]
soup_between = BeautifulSoup(between_content, 'html.parser')

if len(profile := soup_between.find_all('div', class_ = 'wiki-table-wrap table-right')) > 0:
    # print(profile[-1].get_text())
    toc_dict['s-p'] = ('profile', profile[-1]) # 마지막엔 각주 영역

# 딕셔너리 형태로 저장(key = s-#.#.#, value = (목차 명, 목차 element))
for i, e in enumerate(toc.find_all("span", class_  = "toc-item")):
    toc_dict[e.find('a')['href'].replace("#", "")] = (e.get_text(), soup.find('a', id=e.find('a')['href'].replace("#", "")))
toc_dict['s-f'] = ('footnote', soup.find("div", class_ = 'wiki-macro-footnote')) # 마지막엔 각주 영역

# 목차 프린트
print(f"\nDocument of {soup.find("a", href = "/w/"+url.split("/w/")[-1].split("?")[0]).get_text()}\n")
print("\n================== Table of Contents  ==================\n")
for k, v in toc_dict.items():
    num = k.count(".")
    print("".join(["\t"] * num) + v[0])


def get_content_between_heandings(soup, start_tag, end_tag):
    # HTML 콘텐츠를 문자열로 변환
    html_str = str(soup)

    # 시작 태그와 끝 태그의 위치를 찾기
    start_pos = html_str.find(str(start_tag)) + len(str(start_tag))
    end_pos = html_str.find(str(end_tag))

    # 시작 태그와 끝 태그 사이의 콘텐츠를 추출
    between_content = html_str[start_pos:end_pos]

    # 추출한 콘텐츠를 다시 파싱
    soup_between = BeautifulSoup(between_content, 'html.parser')

    # 특정 클래스명을 가진 엘리먼트를 수집
    elements_between = soup_between.find_all('div', class_='wiki-paragraph')

    # 결과 출력
    for element in elements_between:
        # print(element)
        print(element.get_text())

def get_item_and_next(dictionary, target_key):
    keys = list(dictionary.keys())
    values = list(dictionary.values())
    
    if target_key in dictionary:
        index = keys.index(target_key)
        if index < len(keys) - 1:
            return keys[index + 1], values[index + 1]
        else:
            return {keys[index]: values[index]}  # 마지막 키인 경우 다음 아이템이 없음
    else:
        return None

def get_content_heading(soup, toc_dict, heading_idx):
    start_tag = toc_dict.get(heading_idx)[1]
    end_tag = get_item_and_next(toc_dict, heading_idx)[1][1] 
    content = get_content_between_heandings(soup, start_tag= start_tag, end_tag= end_tag)
    return content

