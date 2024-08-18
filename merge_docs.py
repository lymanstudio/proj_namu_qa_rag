from namu_loader import NamuLoader

url = 'https://namu.wiki/w/ILLIT'
# url = 'https://namu.wiki/w/%EB%B0%95%EC%84%B1%EC%88%98(%EC%A0%95%EC%B9%98%EC%9D%B8)'


loader = NamuLoader(url, 1, verbose = True)
docs = loader.load()

get_toc_item_depth = lambda item: sum(len(frag.split('/')) for frag in item.split('//'))


def print_docs(docs):
    for doc in docs:
        print(f"metadata: {doc.metadata}, contents: {doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content}")

print_docs(docs)

# for i, doc in docs:



import pandas as pd

def export_docs_to_csv(docs):
    df = pd.DataFrame(columns = list(docs[0].metadata.keys()) + ['page_contents'])

    for i, doc in enumerate(docs):
        df.loc[i] = list(doc.metadata.values()) + [doc.page_content]

    df.to_csv('./temp.csv')
    return df

df = export_docs_to_csv(docs)