import os
from typing import Optional
from llama_parse import LlamaParse
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq 
from llama_index.vector_stores.kdbai import KDBAIVectorStore
import kdbai_client as kdbai
from dotenv import load_dotenv
# Environment Configuration
load_dotenv()
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HF_TOKEN")
os.environ["LLAMA_CLOUD_API_KEY"] = os.getenv("LLAMA_CLOUD_API_KEY")
class PDFRAGSystem:
    _DEFAULT_PARSING_INSTRUCTIONS = (
        "Extract all text, tables, and figures faithfully. "
        "Preserve numerical data and structural relationships. "
        "Pay special attention to separating individual names with first name and last name; if names appear in a list or line, "
        "ensure each name is captured as a separate text chunk."
        "Maintain complete sentences and contextual relationships between paragraphs."
    )
    _EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
    _LLM_MODEL_NAME = "mixtral-8x7b-32768" 
    _TABLE_NAME = "LlamaParse_RAG_Table2"
    _VECTOR_INDEX_CONFIG = {
        "name": "flat",
        "type": "flat",
        "column": "embeddings",
        "params": {"dims": 768, "metric": "CS"}
    }
    def __init__(self):
        """Initialize the RAG system with environment configuration"""
        self._validate_environment()
        self._initialize_models()
        self._initialize_vector_db()
    def _validate_environment(self):
        """Ensure required environment variables are set"""
        required_vars = ["GROQ_API_KEY", "LLAMA_CLOUD_API_KEY", "KDB_API_KEY"]
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            raise ValueError(f"Missing environment variables: {', '.join(missing)}")
    def _initialize_models(self):
        """Configure AI models with shared settings"""
        self.embed_model = HuggingFaceEmbedding(model_name=self._EMBED_MODEL_NAME)
        self.llm = Groq(
            model=self._LLM_MODEL_NAME,
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.1,
            max_tokens=512
        )
        Settings.embed_model = self.embed_model
        Settings.llm = self.llm
    def _initialize_vector_db(self):
        """Set up KDB.AI connection and reset the table to avoid context mixing"""
        self.session = kdbai.Session(
            api_key=os.getenv("KDB_API_KEY"),
            endpoint=os.getenv("KDBAI_ENDPOINT", "https://cloud.kdb.ai/instance/i6boonn29w")
        )
        self.db = self.session.database("default")
        self._reset_vector_store()
    def _reset_vector_store(self):
        """Drop the existing table and recreate it to prevent old document mixing"""
        try:
            self.db.table(self._TABLE_NAME).drop()
        except kdbai.KDBAIException as e:
            if "not found" not in str(e).lower():
                raise RuntimeError(f"Failed to reset vector store: {str(e)}")
        self.table = self.db.create_table(
            self._TABLE_NAME,
            schema=[
                {"name": "document_id", "type": "str"},
                {"name": "text", "type": "str"},
                {"name": "embeddings", "type": "float32s"},
            ],
            indexes=[self._VECTOR_INDEX_CONFIG]
        )
    def process_pdf(self, pdf_path: str, parsing_instructions: Optional[str] = None):
        """Process and index a PDF document"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        self._reset_vector_store()
        parser = LlamaParse(
            result_type="markdown",
            content_guideline_instruction=parsing_instructions or self._DEFAULT_PARSING_INSTRUCTIONS,
            num_workers=min(4, os.cpu_count() or 1))
        try:
            documents = parser.load_data(pdf_path)
            node_parser = MarkdownElementNodeParser(num_workers=min(4, os.cpu_count() or 1), chunk_size=512)
            print(f"PDF parsed successfully. Number of documents: {len(documents)}")
            nodes = node_parser.get_nodes_from_documents(documents)
            base_nodes, objects = node_parser.get_nodes_and_objects(nodes)
            total_nodes = len(base_nodes) + len(objects)
            print(f"Total nodes for indexing: {total_nodes}")
            print(nodes)
            vector_store = KDBAIVectorStore(self.table)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            self.index = VectorStoreIndex(
                nodes=base_nodes + objects,
                storage_context=storage_context,
                show_progress=True
            )
            storage_context.persist()
        except Exception as e:
            raise RuntimeError(f"PDF processing failed: {str(e)}")
    # def _fallback_response(self, question: str, context: str) -> str:
    #     """Handle incomplete responses with alternative strategies"""
    #     retry_prompt = f"Re-analyze this context and provide a more comprehensive answer to: {question}\n\nContext: {context[:10000]}"
    #     return self.llm.complete(retry_prompt).text.strip()
    def query(self, question: str, top_k: int = 20) -> str:
        """Execute a query against the RAG system"""
        try:
            query_embedding = self.embed_model.get_text_embedding(question)
            # Search for top_k matches
            results = self.table.search(vectors={"flat": [query_embedding]}, n=top_k)
            # Aggregate context from results
            if isinstance(results, list):
                all_rows = []
                for res in results:
                    all_rows.extend(res.to_dict(orient="records"))
                context = "\n".join([row["text"][:5000] for row in all_rows])
            else:
                context = "\n".join(results["text"].tolist())
            # Build the prompt with full context from all matching nodes
            prompt = self._format_prompt(context, question)
            response = self.llm.complete(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Query failed: {str(e)}"
    def _format_prompt(self, context: str, question: str) -> str:
        """Generate structured prompt template"""
        return (
            f"<s>[INST] <<SYS>>\n"
            f"You are a technical research assistant. Use ONLY the context below to answer the question.\n"
            f"Be concise, factual, and reference specific parts of the context where possible.\n"
            f"If the answer isn't found in the context, simply say \"I don't know\".\n"
            f"Please provide a complete and detailed answer\n"
            f"Always complete the answer with a fullstop\n"
            f"<</SYS>>\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            f"Answer: [/INST]"
        )
if __name__ == "__main__":
    try:
        rag_system = PDFRAGSystem()
        # # For Colab users, upload a file via the widget. Otherwise, provide a PDF file path.
        # try:
        #     from google.colab import files
        #     uploaded = files.upload()
        #     pdf_path = next(iter(uploaded)) if uploaded else "default.pdf"
        # except ImportError:
        #     pdf_path = "default.pdf"
        # rag_system.process_pdf(pdf_path)
        # # Change your query to be more specific if needed
        # question = "List all the names found in the council tax application."
        # print(f"\nQuestion: {question}")
        # print(f"Answer: {rag_system.query(question)}")
    except Exception as e:
        print(f"System initialization failed: {str(e)}")