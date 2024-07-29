import os
import json
import time
from namu_loader import NamuLoader
import textwrap

from google.cloud import bigquery

import openai
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_google_community import BigQueryVectorStore

# BigQuery
def load_data_to_bigquery(client, json_data, project_id, dataset_id, table_id, region, artist_info, page_url, write_disposition):
    
    # metadata 는 한글이 섞여있으므로 ensure_ascii 옵션을 False 로 설정한다.
    # artist_info, page_url 은 크롤링된 정보에서 가져오는 것이 아니므로 수동으로 넣어준다.
    for item in json_data:
        item['metadata'] = json.dumps(item['metadata'], ensure_ascii=False)
        item['artist_info'] = artist_info
        item['page_url'] = page_url
    
    table_ref = client.dataset(dataset_id, project=project_id).table(table_id)
    job_config = bigquery.LoadJobConfig()
    job_config.write_disposition = write_disposition
    
    load_job = client.load_table_from_json(
        json_data, table_ref, location=region, job_config=job_config
    )
    
    load_job.result()  
    print(f'Loaded {len(json_data)} rows into {project_id}:{dataset_id}.{table_id}')

def main():

	  # NamuLoader 를 사용해서 url 정보를 크롤링한다.
    url = 'https://namu.wiki/w/ILLIT'
    max_hop = 1
    verbose = True
    loader = NamuLoader(url=url, max_hop=max_hop, verbose=verbose)

		# 크롤링한 데이터를 documents 에 append 
    documents = []
    for doc in loader.lazy_load():
        documents.append({
            "page_content": doc.page_content,
            "metadata": doc.metadata
        })
    
    # 내가 작업하고자 하는 GCP 프로젝트, region, dataset, table id 설정
    PROJECT_ID = "wev-dev-analytics"
    REGION = "asia-northeast3"
    DATASET_ID = "namu_wiki"
    TABLE_ID = "artists"

    # 빅쿼리에 저장할 테이블의 schema 정의
    client = bigquery.Client(project=PROJECT_ID)
    schema = [
      bigquery.SchemaField("artist_info", "STRING"),
      bigquery.SchemaField("page_url", "STRING"),
      bigquery.SchemaField("page_content", "STRING"),
      bigquery.SchemaField("metadata", "STRING")
      ]
    
    dataset_ref = client.dataset(DATASET_ID)
    dataset = bigquery.Dataset(dataset_ref)
    dataset.location = REGION

    # 데이터셋 생성 (이미 존재하는 경우 생략)
    try:
        client.create_dataset(dataset)
        print(f"Created dataset {DATASET_ID} in {REGION}")
    except:
        print(f"Dataset {DATASET_ID} already exists in {REGION}")
    
    # 테이블 생성 (이미 존재하는 경우 생략)
    table_ref = dataset_ref.table(TABLE_ID)
    table = bigquery.Table(table_ref, schema=schema)

    try:
        client.create_table(table)
        print(f"Created table {TABLE_ID} in dataset {DATASET_ID}")
    except:
        print(f"Table {TABLE_ID} already exists in dataset {DATASET_ID}")

		# 넣고 싶은 ARTIST_INFO, PAGE_URL 값을 기입해준다.
    ARTIST_INFO = loader.base_nc.get_doc_title()
    PAGE_URL = url

		# 각 파라미터를 기입해준다. WRITE_APPEND 은 테이블에 데이터가 append 되고, WRITE_TRUNCATE 을 기입하면 overwrite 된다.
    load_data_to_bigquery(client, documents, PROJECT_ID, DATASET_ID, TABLE_ID, REGION, ARTIST_INFO, PAGE_URL, bigquery.WriteDisposition.WRITE_APPEND) # WRITE_APPEND, WRITE_TRUNCATE

if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Elapsed time: {round(time.time() - start_time, 2)} seconds")

