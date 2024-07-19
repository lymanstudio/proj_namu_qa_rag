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

from namu_loader import NamuLoader

cur_notebook_dir = os.getcwd()
os.chdir("../")
base_dir = os.getcwd()
key_dir = os.path.join(base_dir, 'keys')
os.chdir(base_dir)
print(os.getcwd())

from dotenv import load_dotenv
print(load_dotenv(dotenv_path= os.path.join(key_dir, ".env")))
os.chdir(cur_notebook_dir)

def get_documnet_from_namuwiki(url, NamuLoader_obj, hop = 1):
    loader = NamuLoader_obj(url, hop, verbose = True)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
        )
    splitDocs = splitter.split_documents(docs)
    return splitDocs

def create_db(docs):
    embedding = OpenAIEmbeddings()
    vectorStore = FAISS.from_documents(docs, embedding = embedding)
    return vectorStore

def get_chain(vectorStore, model):
    prompt = ChatPromptTemplate.from_template("""
        Answer the users's QUESTION regarding CONTEXT. Use the language that the user types in.
        If you can not find an appropriate answer, return "No Answer found.".
        CONTEXT: {context}
        QUESTION: {question}
    """)

    retriever = vectorStore.as_retriever()

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

#============================================= RUN =============================================

url = 'https://namu.wiki/w/ILLIT'
# url = 'https://namu.wiki/w/%EB%B0%95%EC%84%B1%EC%88%98(%EC%A0%95%EC%B9%98%EC%9D%B8)'

docs = get_documnet_from_namuwiki(url = url , NamuLoader_obj = NamuLoader, hop = 1)

def print_docs(docs):
    for doc in docs:
        print(f"metadata: {doc.metadata}, contents: {doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content}")

print_docs(docs)


vectorStore = create_db(docs)
chain = get_chain(vectorStore, llm)
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


question = "민주에 대해 알려줘"
get_answer(chain, question)