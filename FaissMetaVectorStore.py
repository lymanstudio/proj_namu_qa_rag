
from __future__ import annotations

import logging
import operator
import os
import pickle
import uuid
import warnings
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Union,
    TypeVar,
    Type
)

VST = TypeVar("VST", bound="VectorStore")

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables.config import run_in_executor
from langchain_core.vectorstores import VectorStore

from langchain_community.docstore.base import AddableMixin, Docstore
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores.utils import (
    DistanceStrategy,
    maximal_marginal_relevance,
)
from langchain_community.vectorstores.faiss import FAISS, dependable_faiss_import, _len_check_if_sized
logger = logging.getLogger(__name__)

def concat_selected_fields_in_metadatas(
        metadatas: List[dict] = None,
        selected_fields: List[str] = None,
        separator: Optional[str] = ", " 
    ) -> List[str]:
    
    common_keys = set(metadatas[0].keys())
    for metadata in metadatas[1:]:
        common_keys.intersection_update(metadata.keys())
    
    if all(field in common_keys for field in selected_fields) == False:
        raise KeyError(
            f"One or more keys in the metadata_fields is not in the metadata dictionaries"
        )
    
    return [separator.join([str(metadata.get(field)) for field in selected_fields]) for metadata in metadatas]
    
class FaissMetaVectorStore(FAISS):
    def __init__(
            self,
            embedding_function: Union[
                Callable[[str], List[float]],
                Embeddings,
            ],
            index: Any,
            metaindex: Any,
            content_docstore: Docstore,
            metadata_docstore: Docstore,
            index_to_docstore_id: Dict[int, str],
            index_to_metadata_docstore_id: Dict[int, str],
            relevance_score_fn: Optional[Callable[[float], float]] = None,
            normalize_L2: bool = False,
            distance_strategy: DistanceStrategy = DistanceStrategy.EUCLIDEAN_DISTANCE,
        ):
        """Initialize with necessary components."""
        if not isinstance(embedding_function, Embeddings):
            logger.warning(
                "`embedding_function` is expected to be an Embeddings object, support "
                "for passing in a function will soon be removed."
            )
        self.embedding_function = embedding_function
        self.index = index
        self.metaindex = metaindex
        self.content_docstore = content_docstore
        self.metadata_docstore = metadata_docstore
        self.index_to_docstore_id = index_to_docstore_id
        self.index_to_metadata_docstore_id = index_to_metadata_docstore_id
        self.distance_strategy = distance_strategy
        self.override_relevance_score_fn = relevance_score_fn
        self._normalize_L2 = normalize_L2
        if (
            self.distance_strategy != DistanceStrategy.EUCLIDEAN_DISTANCE
            and self._normalize_L2
        ):
            warnings.warn(
                "Normalizing L2 is not applicable for "
                f"metric type: {self.distance_strategy}"
            )

    def __add(
            self,
            texts: Iterable[str],
            embeddings: Iterable[List[float]],
            docstore: Docstore,
            index_to_docstore_id: Dict[int, str],
            index: Any,
            metadatas: Optional[Iterable[dict]] = None,
            ids: Optional[List[str]] = None,
        ) -> List[str]:
        faiss = dependable_faiss_import()

        if not isinstance(docstore, AddableMixin):
            raise ValueError(
                "If trying to add texts, the underlying docstore should support "
                f"adding items, which {docstore} does not"
            )

        # texts와 metadatas의 사이즈(개수)가 같은지 체크
        _len_check_if_sized(texts, metadatas, "texts", "metadatas") 
        _metadatas = metadatas or ({} for _ in texts)
        
        # 다시 Document들을 만들어줌
        documents = [
            Document(page_content=t, metadata=m) for t, m in zip(texts, _metadatas)
        ]

        # documents와 embeddings, ids의 사이즈(개수)가 같은지 체크
        _len_check_if_sized(documents, embeddings, "documents", "embeddings") 
        _len_check_if_sized(documents, ids, "documents", "ids")
        
        # ids 들이 중복이 있는지 체크
        if ids and len(ids) != len(set(ids)):
            raise ValueError("Duplicate ids found in the ids list.")

        # Add to the index. ## 실제 임베딩 벡터들을 index라는 곳에 저장하는 단계
        vector = np.array(embeddings, dtype=np.float32)
        if self._normalize_L2:
            faiss.normalize_L2(vector)
        index.add(vector)

        # Add information to docstore and index. ## Document와 추가 정보들을 저장하는 단계
        ids = ids or [str(uuid.uuid4()) for _ in texts]
        docstore.add({id_: doc for id_, doc in zip(ids, documents)})
        starting_len = len(index_to_docstore_id)
        index_to_id = {starting_len + j: id_ for j, id_ in enumerate(ids)}
        index_to_docstore_id.update(index_to_id)

        return ids

    @classmethod
    def __from(
            cls,
            texts: Iterable[str],
            metadata_texts: Iterable[str],
            content_embeddings: List[List[float]],
            metadata_embeddings: List[List[float]],
            embedding: Embeddings,
            metadatas: Optional[Iterable[dict]] = None,
            ids: Optional[List[str]] = None,
            normalize_L2: bool = False,
            distance_strategy: DistanceStrategy = DistanceStrategy.EUCLIDEAN_DISTANCE,
            **kwargs: Any,
        ) -> FaissMetaVectorStore:
        faiss = dependable_faiss_import()
        if distance_strategy == DistanceStrategy.MAX_INNER_PRODUCT:
            index = faiss.IndexFlatIP(len(content_embeddings[0]))
            metaindex = faiss.IndexFlatIP(len(content_embeddings[0]))
        else:
            # Default to L2, currently other metric types not initialized.
            index = faiss.IndexFlatL2(len(content_embeddings[0]))
            metaindex = faiss.IndexFlatL2(len(content_embeddings[0]))
        
        content_docstore = kwargs.pop("docstore", InMemoryDocstore())
        metadata_docstore = kwargs.pop("docstore", InMemoryDocstore())
        index_to_docstore_id = kwargs.pop("index_to_docstore_id", {})
        index_to_metadata_docstore_id = kwargs.pop("index_to_docstore_id", {})

        vecstore = cls(
            embedding,
            index,
            metaindex,
            content_docstore,
            metadata_docstore,
            index_to_docstore_id,
            index_to_metadata_docstore_id,
            normalize_L2=normalize_L2,
            distance_strategy=distance_strategy,
            **kwargs,
        )

        ids = vecstore.__add(
            texts, 
            embeddings = content_embeddings, 
            docstore = vecstore.content_docstore,
            index_to_docstore_id = vecstore.index_to_docstore_id,
            index = vecstore.index,
            metadatas=metadatas, 
            ids=ids
        )
        _ = vecstore.__add(
            metadata_texts, 
            embeddings = metadata_embeddings, 
            docstore = vecstore.metadata_docstore,
            index_to_docstore_id = vecstore.index_to_metadata_docstore_id,
            index = vecstore.metaindex,
            metadatas=metadatas, 
            ids=ids
        )
        return vecstore
    
    @classmethod
    def from_texts(cls,
            texts: List[str],
            embedding: Embeddings,
            metadatas: List[dict] = None,
            metadata_fields: List[str] = None,
            ids: Optional[List[str]] = None,
            **kwargs: Any,
        ) -> FaissMetaVectorStore:
        
        content_embeddings = embedding.embed_documents(texts)
        
        if metadata_fields is not None:
            metadata_texts = concat_selected_fields_in_metadatas(metadatas=metadatas, selected_fields = metadata_fields)
            metadata_embeddings = embedding.embed_documents(metadata_texts)

        return cls.__from(
            texts,
            metadata_texts = metadata_texts,
            content_embeddings=content_embeddings,
            metadata_embeddings=metadata_embeddings,
            embedding=embedding,
            metadatas=metadatas,
            ids=ids, 
            **kwargs, 
        )
    
        
    @classmethod
    def from_documents(
        cls: Type[VST],
        documents: List[Document],
        metadata_fields: List[str],
        embedding: Embeddings,
        **kwargs: Any,
    ) -> VST:
        """Return VectorStore initialized from documents and embeddings.

        Args:
            documents: List of Documents to add to the vectorstore.
            embedding: Embedding function to use.
            kwargs: Additional keyword arguments.

        Returns:
            VectorStore: VectorStore initialized from documents and embeddings.
        """
        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]
        return cls.from_texts(texts, embedding, metadatas=metadatas, metadata_fields = metadata_fields, **kwargs)