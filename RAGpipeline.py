from typing import Optional
from llama_parse import LlamaParse
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceInferenceAPI
from llama_index.vector_stores.kdbai import KDBAIVectorStore
import kdbai_client as kdbai
from dotenv import load_dotenv
import os

# Environment Configuration
load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HF_TOKEN")
os.environ["LLAMA_CLOUD_API_KEY"] = os.getenv("LLAMA_CLOUD_API_KEY")

class PDFRAGSystem:
    _DEFAULT_PARSING_INSTRUCTIONS = """Extract all text, tables, and figures faithfully.
                                    Preserve numerical data and structural relationships."""
    _EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
    _LLM_MODEL_NAME = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    _TABLE_NAME = "LlamaParse_RAG_Table2"
    _VECTOR_INDEX_CONFIG = {
        "name": "flat",
        "type": "flat",
        "column": "embeddings",
        "params": {"dims": 768, "metric": "L2"}
    }

    def __init__(self):
        """Initialize the RAG system with environment configuration"""
        self._validate_environment()
        self._initialize_models()
        self._initialize_vector_db()

    def _validate_environment(self):
        """Ensure required environment variables are set"""
        required_vars = ["HF_TOKEN", "LLAMA_CLOUD_API_KEY", "KDB_API_KEY"]
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            raise ValueError(f"Missing environment variables: {', '.join(missing)}")

    def _initialize_models(self):
        """Configure AI models with shared settings"""
        self.embed_model = HuggingFaceEmbedding(model_name=self._EMBED_MODEL_NAME)
        self.llm = HuggingFaceInferenceAPI(
            model_name=self._LLM_MODEL_NAME,
            token=os.getenv("HF_TOKEN")
        )
        Settings.embed_model = self.embed_model
        Settings.llm = self.llm

    def _initialize_vector_db(self):
        """Set up KDB.AI connection and table"""
        self.session = kdbai.Session(
            api_key=os.getenv("KDB_API_KEY"),
            endpoint=os.getenv("KDBAI_ENDPOINT", "https://cloud.kdb.ai/instance/i6boonn29w")
        )
        self.db = self.session.database("default")
        self.table = self._create_kdbai_table()

    def _create_kdbai_table(self) -> kdbai.Table:
        """Initialize or reset the vector database table"""
        schema = [
            {"name": "document_id", "type": "str"},
            {"name": "text", "type": "str"},
            {"name": "embeddings", "type": "float32s"},
        ]

        try:
            # Attempt to delete existing table
            self.db.table(self._TABLE_NAME).drop()
        except kdbai.KDBAIException as e:
            # Ignore "Table not found" errors
            if "not found" not in str(e).lower():
                raise RuntimeError(f"Failed to initialize KDB.AI table: {str(e)}")

        return self.db.create_table(
            self._TABLE_NAME,
            schema,
            indexes=[self._VECTOR_INDEX_CONFIG]
        )

    def process_pdf(self, pdf_path: str, parsing_instructions: Optional[str] = None):
        """Process and index a PDF document"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        parser = LlamaParse(
            result_type="markdown",
            content_guideline_instruction=parsing_instructions or self._DEFAULT_PARSING_INSTRUCTIONS,
            num_workers=min(4, os.cpu_count() or 1)  # Adaptive worker count
        )
        
        try:
            documents = parser.load_data(pdf_path)
            node_parser = MarkdownElementNodeParser(num_workers=min(4, os.cpu_count() or 1))
            nodes = node_parser.get_nodes_from_documents(documents)
            base_nodes, objects = node_parser.get_nodes_and_objects(nodes)

            vector_store = KDBAIVectorStore(self.table)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            VectorStoreIndex(
                nodes=base_nodes + objects,
                storage_context=storage_context,
                show_progress=True
            )
        except Exception as e:
            raise RuntimeError(f"PDF processing failed: {str(e)}")

    def query(self, question: str, top_k: int = 5) -> str:
        """Execute a query against the RAG system"""
        try:
            query_embedding = self.embed_model.get_text_embedding(question)
            results = self.table.search(
                vectors={"flat": [query_embedding]},
                n=top_k,
                filter=[("<>", "document_id", "4a9551df-5dec-4410-90bb-43d17d722918")]
            )
            context = "\n".join([row["text"] for _, row in results[0].iterrows()])

            response = self.llm.complete(self._format_prompt(context, question))
            return response.text.strip()
        except Exception as e:
            return f"Query failed: {str(e)}"

    def _format_prompt(self, context: str, question: str) -> str:
        """Generate structured prompt template"""
        return f"""<s>[INST] <<SYS>>
        You are a technical research assistant. Answer the question using ONLY the context below.
        Be concise and factual. If the answer isn't in the context, say "I don't know".
        <</SYS>>

        Context:
        {context}

        Question: {question}
        Answer: [/INST]"""

if __name__ == "__main__":
    try:
        rag_system = PDFRAGSystem()
        
        # For Colab users
        from google.colab import files
        uploaded = files.upload()
        pdf_path = next(iter(uploaded)) if uploaded else "default.pdf"
        rag_system.process_pdf(pdf_path)

        question = "What was the main finding of the study regarding prompt variations?"
        print(f"\nQuestion: {question}")
        print(f"Answer: {rag_system.query(question)}")
    except Exception as e:
        print(f"System initialization failed: {str(e)}")