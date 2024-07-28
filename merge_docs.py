from namu_loader import NamuLoader

url = 'https://namu.wiki/w/ILLIT'
# url = 'https://namu.wiki/w/%EB%B0%95%EC%84%B1%EC%88%98(%EC%A0%95%EC%B9%98%EC%9D%B8)'


loader = NamuLoader_obj(url, hop, verbose = True)
docs = loader.load()