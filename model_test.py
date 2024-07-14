from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import ConfigurableField
from operator import itemgetter
import argparse
import os

from namu_loader import NamuLoader

cur_notebook_dir = os.getcwd()
base_dir = os.getcwd()
key_dir = os.path.join(base_dir, 'keys')
data_dir = os.path.join(base_dir, 'data')
post_dir = os.path.join(data_dir, 'tistory_post')
sample_dir = os.path.join(data_dir, 'samples')
os.chdir(base_dir)
print(os.getcwd())

from dotenv import load_dotenv
print(load_dotenv(dotenv_path= os.path.join(key_dir, ".env")))
os.chdir(cur_notebook_dir)


llm = (
    ChatAnthropic(model_name='claude-3-opus-20240229')
    .configurable_alternatives(
        ConfigurableField(id = 'llm'),
        default_key="claude3-opus",
        claude3_haiku=ChatAnthropic(model_name='claude-3-haiku-20240307'),
        gpt4o = ChatOpenAI(model = 'gpt-4o'),
        gpt3_5 = ChatOpenAI(model = 'gpt-3.5-turbo'),
    )
    .configurable_fields(
        temperature=ConfigurableField(
            id="temperature",
            name="LLM Temperature",
            description="The temperature of the LLM",
        )
    )
)

#================================================================================================

# url = 'https://namu.wiki/w/ILLIT/Weverse'
url = 'https://namu.wiki/w/%EA%B5%AD%EA%B0%80%EA%B3%B5%EC%9D%B8%20%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%B6%84%EC%84%9D%20%EC%A0%84%EB%AC%B8%EA%B0%80'

loader = NamuLoader(url = url, max_hop = 0, verbose= True)

docs = loader.load()

def print_docs(docs):
    for doc in docs:
        print(f"metadata: {doc.metadata}, contents: {doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content}")

print_docs(docs)