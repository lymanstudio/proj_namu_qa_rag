import argparse
import os
from operator import itemgetter
from time import time

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import ConfigurableField
from langchain_community.vectorstores.faiss import FAISS

from FaissMetaVectorStore import FaissMetaVectorStore

from namu_loader import NamuLoader

cur_notebook_dir = "C:\\Users\\thdwo\\Documents\\Github\\proj_artist_info_gen"#os.getcwd()
os.chdir(cur_notebook_dir)
os.chdir("../")
base_dir = os.getcwd()
key_dir = os.path.join(base_dir, 'keys')
os.chdir(base_dir)
print(os.getcwd())

def export_csv(docs):
    import pandas as pd

    tmp_df = pd.DataFrame(columns = list(docs[0].metadata.keys()) + ['page_content'])
    for doc in docs:
        if len(doc.page_content) > 200:
            page_content = doc.page_content[:200]
        else:
            page_content = doc.page_content
        cur_data = doc.metadata
        cur_data['page_content']= page_content
        tmp_df = pd.concat([tmp_df, pd.DataFrame([cur_data])], ignore_index=True)
    return tmp_df

from dotenv import load_dotenv
print(load_dotenv(dotenv_path= os.path.join(key_dir, ".env")))
os.chdir(cur_notebook_dir)

def get_documnet_from_namuwiki(url, NamuLoader_obj, hop = 1):
    loader = NamuLoader_obj(url, hop, verbose = True)
    docs = loader.load()

    # splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=500,
    #     chunk_overlap=20
    #     )
    # docs = splitter.split_documents(docs)
    return docs


embedding = OpenAIEmbeddings()

def get_chain(retriever, model):
    prompt = ChatPromptTemplate.from_template("""
        Answer the users's QUESTION regarding CONTEXT. Use the language that the user types in.
        If you can not find an appropriate answer, return "No Answer found.".
        CONTEXT: {context}
        QUESTION: {question}
    """)

    def concat_docs(docs) -> str: # retriever가 반환한 모든 Document들의 page_content를 하나의 단일 string으로 붙여주는 함수
        return "".join(doc.page_content for doc in docs)

    chain = {
        "context" : itemgetter('question') | retriever | concat_docs
        , "question" : itemgetter('question') | RunnablePassthrough()
    } | prompt| model | StrOutputParser()

    return chain

llm = (
    ChatAnthropic(model_name='claude-3-5-sonnet-20240620')
    .configurable_alternatives(
        ConfigurableField(
            id = 'llm',
            name="LLM Model",
            description="The base LLM model",
        ),
        default_key="claude3_5_sonnet",
        claude3_haiku=ChatAnthropic(model_name='claude-3-haiku-20240307'),
        gpt4o = ChatOpenAI(model = 'gpt-4o'),
        gpt3_5 = ChatOpenAI(model = 'gpt-3.5-turbo'),
    )
    .configurable_fields(
        temperature=ConfigurableField(
            id="temperature",
            name="LLM Temperature",
            description="The temperature of the LLM",
        ),
        max_tokens = ConfigurableField(
            id="max_token",
            name="Maximum input Tokens",
            description="Maximum limit of input Tokens",
        ),
    )
)

def get_answer(chain, question,  temp = .1, max_tokens = 2048):
    for model_name in ['claude3_haiku', 'claude3_5_sonnet', 'gpt3_5', 'gpt4o']:
        try:
            start = time()
            result = chain.with_config(configurable={
                        "llm": model_name, 
                        "temparature": temp,
                        "max_tokens": max_tokens
                    }).invoke({"question": question})
            print("\n\nModel: {} \t\t Answer: {} \t\t elapsed_time: {}".format(model_name, result, round(time() - start, 1)))
        except:
            print("\n\nModel: {} \t\t Answer: {}".format(model_name, "Error."))


#============================================= RUN =============================================

url = 'https://namu.wiki/w/ILLIT'
url = 'https://namu.wiki/w/%EC%84%9C%EC%9A%B8%20%EB%B2%84%EC%8A%A4%205413'
# url = 'https://namu.wiki/w/%EB%B0%95%EC%84%B1%EC%88%98(%EC%A0%95%EC%B9%98%EC%9D%B8)'

docs = get_documnet_from_namuwiki(url = url , NamuLoader_obj = NamuLoader, hop = 0)

def print_docs(docs):
    for doc in docs:
        print(f"metadata: {doc.metadata}, contents: {doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content}")

print_docs(docs)


k = 10

vectorStore = FAISS.from_documents(docs, embedding = embedding)
vs_meta = FaissMetaVectorStore.from_documents(docs, embedding = embedding, metadata_fields= ["abs_page_toc_item", "base_page_url"])

ret = vectorStore.as_retriever(search_kwargs={'k': k})
chain = get_chain(ret, llm)

question = "아일릿의 앨범에 대해 알려줘"

get_answer(chain, question)




#================================================Ensemble Retriever=============================================================
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

# from langchain_core.callbacks import CallbackManagerForRetrieverRun

# docs_meta = [Document(page_content = doc.metadata['abs_page_toc_item'], metadata = doc.metadata) for doc in docs]

vectorStore = FAISS.from_documents(docs, embedding = embedding)
vectorStore_meta = FAISS.from_texts([doc.metadata['abs_page_toc_item'] for doc in docs], embedding, metadatas= [doc.metadata for doc in docs])
k = 10
ret_content = vectorStore.as_retriever(search_kwargs={'k': k})
ret_meta = vectorStore_meta.as_retriever(search_kwargs={'k': k})
ret_keyword = BM25Retriever.from_documents([doc for doc in vectorStore_meta.docstore._dict.values()], search_kwargs={'k': k})

# get_orig_doc = lambda metadoc: [doc for doc in docs if doc.metadata['abs_page_toc_item'] == metadoc.metadata['abs_page_toc_item']]

from typing import List, Optional
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever, RetrieverLike

class MetadataRetriever(EnsembleRetriever):
    retrievers: List[RetrieverLike]
    weights: List[float]
    c: int = 60
    id_key: Optional[str] = None
    orig_docs: List[Document]

    def get_docs(self, res_docs):
        get_orig_doc = lambda metadoc: [doc for doc in self.orig_docs if doc.metadata['abs_page_toc_item'] == metadoc.metadata['abs_page_toc_item']]
        ret_docs = []
        print("dd")
        for doc in res_docs: 
            if doc.page_content == doc.metadata['abs_page_toc_item']:
                print("d")
                doc = get_orig_doc(doc)
            if doc not in ret_docs:
                ret_docs.append(doc)
        return ret_docs

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """
        Get the relevant documents for a given query.

        Args:
            query: The query to search for.

        Returns:
            A list of reranked documents.
        """

        # Get fused result of the retrievers.
        fused_documents = self.rank_fusion(query, run_manager)
        print('dfdf')
        return self.get_docs(fused_documents)
    



ensemble_retriever = EnsembleRetriever(
    retrievers=[ret_meta, ret_content], weights=[0.5, 0.5]
)

def get_docs(res_docs, orig_docs):
    get_orig_doc = lambda metadoc: [doc for doc in orig_docs if doc.metadata['abs_page_toc_item'] == metadoc.metadata['abs_page_toc_item']]
    ret_docs = []
    for doc in res_docs: 
        if doc.page_content == doc.metadata['abs_page_toc_item']:
            doc = get_orig_doc(doc)
        if doc not in ret_docs:
            ret_docs.append(doc)
    return ret_docs
        