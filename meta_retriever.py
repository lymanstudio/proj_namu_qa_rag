from langchain_core.documents import Document
from langchain_core.pydantic_v1 import Field
from langchain_core.retrievers import BaseRetriever, RetrieverLike
from typing import List, Optional

from langchain_community.retrievers import BM25Retriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain.retrievers import EnsembleRetriever



class MetadataRetriever(BaseRetriever):
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