import requests
from bs4 import BeautifulSoup
import re

# 웹 페이지 URL
url = 'https://namu.wiki/w/ILLIT'

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



print(soup.find_all("div", class_ = 'wiki-macro-toc')[0].prettify()) ## 목차 division에 있는 모든 컴포넌트 가져옴
print(soup.find_all("a", id = 's-1')[0].prettify()) ## 하이퍼 링크 중 s-1에 해당하는 것들

