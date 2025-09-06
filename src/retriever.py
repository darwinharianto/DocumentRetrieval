from collections import defaultdict
from itertools import chain
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAI


class RetrievalManager:
    """
    Manages the creation and configuration of retrievers and the retrieval chain.
    """

    def __init__(self, vectorstore):
        """
        Initializes the RetrievalManager.

        Args:
            vectorstore: The FAISS vector store instance.
        """
        self.vectorstore = vectorstore
        self.doc_list = self._get_all_documents_from_vectorstore()
        self.ensemble_retriever = None

    def _get_all_documents_from_vectorstore(self):
        """
        Retrieves all documents from the vector store's docstore.
        Note: This is not the most efficient way for large stores, but it's
        simple and effective for this application's scale.
        """
        if not self.vectorstore or not self.vectorstore.docstore:
            return []
        return [
            self.vectorstore.docstore.search(i)
            for i in self.vectorstore.index_to_docstore_id.values()
        ]

    def setup_ensemble_retriever(self, selected_files):
        """
        Sets up the EnsembleRetriever by combining a FAISS retriever and a BM25 retriever.

        Args:
            selected_files (list): A list of source file names to filter the search.
        """
        if not self.doc_list:
            self.ensemble_retriever = None
            return

        # 1. Create FAISS (semantic) retriever
        # Define the number of documents to retrieve
        num_documents_to_retrieve = 4 # Example value, adjust as needed

        faiss_retriever = self.vectorstore.as_retriever(
            search_kwargs={
                "k": num_documents_to_retrieve,
                "filter": {"source": selected_files}
            }
        )

        # 2. Create BM25 (keyword) retriever
        bm25_retriever = BM25Retriever.from_documents(
            self.doc_list, k=num_documents_to_retrieve
        )
        
        # 3. Initialize Ensemble Retriever
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
        )

    def create_retrieval_chain(self):
        """Creates and returns a retrieval chain."""
        if not self.ensemble_retriever:
            return None

        llm = GoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
        
        # This prompt is designed for a question-answering task.
        # It instructs the model on how to use the retrieved context.
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three to five bullet points to keep the answer concise."
            "\n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])

        # The 'create_stuff_documents_chain' is used to format the context (the retrieved documents)
        # into the prompt.
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

        return create_retrieval_chain(self.ensemble_retriever, question_answer_chain)