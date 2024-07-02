import os
from namu_crawler import namuCrawler
base_url = 'https://namu.wiki'
url = 'https://namu.wiki/w/%ED%98%B8%EB%B9%97(%EA%B0%80%EC%9A%B4%EB%8D%B0%EB%95%85)'
url = 'https://namu.wiki/w/%EA%B0%80%EB%82%98%EC%9E%90%EC%99%80%EC%8B%9C'
# url = 'https://namu.wiki/w/%ED%94%84%EB%A1%9C%EB%AF%B8%EC%8A%A4%EB%82%98%EC%9D%B8'
# url = "https://namu.wiki/w/%EA%B4%91%EC%A0%80%EC%9A%B0%20%EC%B0%A8%EC%A7%80"

base_nc = namuCrawler(url = url, hop = 0) ## 타겟 베이스 문서 크롤러 생성
base_nc.construct_toc()
base_nc.print_toc()

def get_total_content(parent_item, url, hop, max_hop):
    sub_nc = namuCrawler(url = url, hop = hop)
    sub_nc.construct_toc()
    # print(sub_nc.get_doc_title(), parent_item, sub_nc.hop, max_hop)
    to_return = ""
    for k, v in sub_nc.toc_dict.items():
        cur_toc_item, content = sub_nc.get_content_heading(k)
        
        if type(content) == str and f'/w/' in content: # content가 링크 대체이면서 
            if sub_nc.hop < max_hop: #현재 문서의 hop이 max_hop보다 적거나 같으면 더 들어가기
                content = get_total_content(parent_item = parent_item, url = base_url + content, hop = sub_nc.hop + 1, max_hop = max_hop)
            else: # max_hop과 같으면 그냥 링크로 대체한다고만 써주기
                content = f"{cur_toc_item[1]}: 다음 문서로 대체 설명: {base_url + content}"
        else: # 일반 설명은 {현재 목차 : 설명} 꼴로 구성
            content = f'{cur_toc_item[1]}: {" ".join(content) if type(content) == list else ""}'
        
        to_return = to_return + "\n" + content + "\n"

    return to_return

def get_a_content(key, max_hop):
    cur_toc_item, content = base_nc.get_content_heading(key)
    if content == None:
        return (cur_toc_item, None)
    elif type(content) == str and '/w/' in content: # content가 링크 대체라면
        return (cur_toc_item, get_total_content(parent_item = base_nc.toc_dict.get(key)[0], url = base_url + content, hop = base_nc.hop + 1, max_hop = max_hop))
    else:
        return (cur_toc_item, " ".join(content))
    

for k, v in base_nc.toc_dict.items():
    print(">> ", k, v[0][1])
    print("\t", get_a_content(k, 2)[1])
 