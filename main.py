import streamlit as st
import openai
import anthropic
import os
import math
from namu_loader import NamuLoader
from get_namu_url import GetNamuUrl
from langchain_openai import OpenAIEmbeddings

chat_gpt_models = {
    "GPT-4o" : "gpt-4o",
    "GPT-4o mini" : "gpt-4o-mini",
    "GPT-4": "gpt-4-0613",
    "GPT-3.5 Turbo": "gpt-3.5-turbo"
}
claude_models = {
    "Claude 3.5 Sonnet" : "claude-3-5-sonnet-20240620",
    "Claude 3 Opus" : "claude-3-opus-20240229", 
    "Claude 3 Sonnet" : "claude-3-sonnet-20240229", 
    "Claude 3 Haiku" : "claude-3-haiku-20240307", 
}

os.environ["GOOGLE_API_KEY"] = "KEY"
os.environ["GOOGLE_SEARCH_ENGINE"] = "KEY"

@st.cache_resource
def is_api_key_valid(api_key, llm_type):
    try:
        if llm_type == 'ChatGPT':
            client = openai.OpenAI(api_key=api_key)
            client.embeddings.create(input = ["Hello"], model="text-embedding-3-small")
        elif llm_type == 'Claude':
            client = anthropic.Anthropic(api_key= api_key)
            result = (
                client
                .messages
                .create(
                    model = "claude-3-haiku-20240307"
                    , max_tokens=1000
                    , temperature=0
                    , system="You are a world-class poet. Respond only with short poems."
                    , messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Hello"
                                }
                            ]
                        }
                    ]
                )
            )
            return True

    except:
        return False
    else:
        return True
    

def main():
    if "ready" not in st.session_state:
        st.session_state.ready = False
    
    status_ok = False

    st.set_page_config(
        page_title = "나무위키 RAG",
        page_icon = ":open_book:"
    )

    st.title("NamuWiki RAG")
    st.write("Find your interest and ask anything about it!")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    with st.sidebar:
        setup_section, namu_doc_section, chain_section = st.tabs(["LLM 세팅", "나무 위키 문서", "상태"])
        
        with setup_section:
            llm_type = st.radio(
                label = 'LLM 타입 선택',
                options = ("ChatGPT", "Claude")
            )
            api_key = st.text_input("API Key", key = "chatbot_api_key", type = "password")
        
            if not api_key:
                st.info(f"Please input your {'OpenAI' if llm_type == 'ChatGPT' else ''} API key to continue.")
                st.stop()

            
            if is_api_key_valid(api_key, llm_type = llm_type) == True:
                st.success("▶ API Key 상태 : {}".format("정상"))
                
                options = chat_gpt_models if llm_type ==  "ChatGPT" else claude_models
                select_model = st.selectbox(
                    label = 'LLM 모델 선택',
                    options = options
                )
                add_k_select =  st.number_input(
                    label = "문서 개수 선택",
                    min_value = 5,
                    max_value =  20,
                    step = 1,
                    key = 'k'
                )
                if llm_type == 'ChatGPT':
                    os.environ['OPENAI_API_KEY'] = api_key
                else:
                    os.environ['ANTHROPIC_API_KEY'] = api_key
                search_type = st.selectbox(
                    label = '검색 타입 선택',
                    options = ("similarity", "mmr", "similarity_score_threshold"),
                )

                if search_type == 'similarity_score_threshold':
                    threshold = st.slider(
                        label = 'Select the threshold',
                        min_value = .5,
                        max_value = 1.0,
                        value = .75,
                        step = .05
                    )
                else:
                    threshold = .75
            else:
                st.error(f"▶ API Key Status: Invaild {'OpenAI' if llm_type == 'ChatGPT' else 'Claude'} API key.")
                st.stop()
            
        with namu_doc_section:
            st.text_input(
                label = "주제어 검색", value = "예: Queen", key = "doc_search_keyword"
            )
            st.toggle("Revision 문서 제외", key = "rev_exclude_flag")
            st.toggle("문서 내 필수 포함 키워드 선택", key = "crucial_keyword_flag", help = "주제어가 포함하고 있을 단어를 입력합니다. 예를 들어, Queen을 검색할 경우 밴드, 여왕, 영국 여왕 등 다양한 주제에 대한 동음이의어가 나올 것인데 필수 포함 키워드로 데뷔를 추가하면 밴드 Queen이, 즉위를 넣으면 영국 여왕이 우선적으로 검색됩니다.")
            if st.session_state.crucial_keyword_flag:
                st.text_input(label= '필수 포함 키워드', value = "예: 데뷔", key = "crucial_keyword")
            
            if st.session_state.doc_search_keyword:
                inst1 = GetNamuUrl(
                    os.getenv("GOOGLE_API_KEY")
                    , os.getenv("GOOGLE_SEARCH_ENGINE")
                    , rev_exclude = st.session_state.rev_exclude_flag
                    , crucial_keyword = st.session_state.crucial_keyword if st.session_state.crucial_keyword_flag else False
                )
                docs = inst1.get_url(st.session_state.doc_search_keyword)
                # st.write(docs)

                if docs == "N/A":
                    st.write("검색 결과 일치하는 문서가 없습니다. 다른 키워드로 검색해주세요.")
                    st.stop()
                doc_dict = dict((doc['title'].split(" - ")[0], doc['formattedUrl']) for doc in docs)
                
                doc_titles = []
                doc_urls = []
                for k, v in doc_dict.items():
                    doc_titles.append(k)
                    doc_urls.append(v)

                st.radio(
                    label = '대상 문서를 선택하세요.',
                    options = doc_titles,
                    captions = doc_urls
                )
            # st.write(st.session_state.doc_search_keyword)






if __name__ == '__main__':
    main()