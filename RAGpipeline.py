import os
import matplotlib.pyplot as plt
import pandas as pd
from PyPDF2 import PdfReader 
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false" 
from typing import Optional
from llama_parse import LlamaParse
from llama_index.core import Settings, StorageContext, VectorStoreIndex
# from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_index.core.node_parser import SimpleNodeParser

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq 
from llama_index.vector_stores.kdbai import KDBAIVectorStore
import kdbai_client as kdbai
from dotenv import load_dotenv
import nest_asyncio
nest_asyncio.apply() 
# Environment Configuration
load_dotenv()
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HF_TOKEN")
os.environ["LLAMA_CLOUD_API_KEY"] = os.getenv("LLAMA_CLOUD_API_KEY")
class PDFRAGSystem:
    _DEFAULT_PARSING_INSTRUCTIONS = (
        "Extract all text content with maximum fidelity. "
        "Preserve original formatting where possible. "
        "Convert visual elements to descriptive text. "
        "Handle ambiguous layouts as linear text flow."
        "Extract all content as linear text with markdown formatting. "
        "Convert tables to markdown format using pipe syntax. "
        "Treat 'Not Provided' as empty values. "
        "Preserve original text order without schema assumptions"
    )
    _EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
    _LLM_MODEL_NAME = "llama-3.3-70b-versatile" 
    _TABLE_NAME = "LlamaParse_RAG_Table2"
    _VECTOR_INDEX_CONFIG = {
        "name": "flat",
        "type": "flat",
        "column": "embeddings",
        "params": {"dims": 768, "metric": "CS"}
    }
    def __init__(self):
        """Initialize the RAG system with environment configuration"""
        try:
            self._validate_environment()
            self._initialize_models()           
            self._initialize_vector_db()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize PDFRAGSystem: {e}")
    def _validate_environment(self):
        """Ensure required environment variables are set"""
        required_vars = ["GROQ_API_KEY", "LLAMA_CLOUD_API_KEY", "KDB_API_KEY"]
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            raise ValueError(f"Missing environment variables: {', '.join(missing)}")
    def _ping_llm(self):
        """Send a minimal request to verify the API key is valid"""  
        try:
            self.llm.complete("", max_tokens=1)
        except Exception as e:
            raise RuntimeError(f"LLM API key validation failed: {e}") 
    def _initialize_models(self):
        """Configure AI models with shared settings"""
        self.embed_model = HuggingFaceEmbedding(model_name=self._EMBED_MODEL_NAME)
        self.llm = Groq(
            model=self._LLM_MODEL_NAME,
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.1,
            max_tokens=1024
        )
        self._ping_llm() 
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
        try:
            reader = PdfReader(pdf_path)
            if len(reader.pages) == 0:
                raise RuntimeError("PDF has zero pages")
        except Exception as e:
            raise RuntimeError(f"PDF processing failed: Malformed PDF ({e})")
        self._reset_vector_store()
        parser = LlamaParse(
            result_type="markdown",
            content_guideline_instruction=parsing_instructions or self._DEFAULT_PARSING_INSTRUCTIONS,
            parse_table=False,
            table_output_format="markdown",
            num_workers=1 #min(4, os.cpu_count() or 1) 
        )
            
        try:
            documents = parser.load_data(pdf_path)
            node_parser = SimpleNodeParser.from_defaults(
                chunk_size=200,
                chunk_overlap=20,
                include_metadata=True
            )
            #node_parser = MarkdownElementNodeParser(num_workers=min(4, os.cpu_count() or 1), chunk_size=1024,include_metadata=True,table_processing_mode="text" )
            print(f"PDF parsed successfully. Number of documents: {len(documents)}")
            nodes = node_parser.get_nodes_from_documents(documents)
            print(nodes)
            # base_nodes, objects = node_parser.get_nodes_and_objects(nodes)
            # total_nodes = len(base_nodes) + len(objects)
            # print(f"Total nodes for indexing: {total_nodes}")
            # print(nodes)
            vector_store = KDBAIVectorStore(self.table)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            self.index = VectorStoreIndex(
                nodes=nodes,
                storage_context=storage_context,
                show_progress=True
            )
            storage_context.persist()
        except Exception as e:
            raise RuntimeError(f"PDF processing failed: {str(e)}")
    def query(self, question: str, top_k: int = 20) -> str:
        """Execute a query against the RAG system"""
        try:
            query_embedding = self.embed_model.get_text_embedding(question)
            results = self.table.search(vectors={"flat": [query_embedding]}, n=top_k)
            MAX_CONTEXT_TOKENS = 6000  
            CHAR_PER_TOKEN = 4
            max_chars = (MAX_CONTEXT_TOKENS // top_k) * CHAR_PER_TOKEN
            if isinstance(results, list):
                all_rows = []
                for res in results:
                    all_rows.extend(res.to_dict(orient="records"))
                context = "\n".join([row["text"][:max_chars] for row in all_rows])
            else:
                context = "\n".join(results["text"].tolist())
            # Build the prompt with full context from all matching nodes
            prompt = self._format_prompt(context, question)
            print("Sending prompt to LLM with key:", os.getenv("GROQ_API_KEY")) 
            response = self.llm.complete(prompt)

            # Inspect for embedded error in the response object
            if hasattr(response, "error") and response.error:   
                raise RuntimeError(f"API returned error: {response.error}")
            if not hasattr(response, "text"):                    
                raise RuntimeError(f"Unexpected API response format: {response}")

            return response.text.strip()

        except Exception as e:
            print("API error encountered in query():", e)          
            return "Sorry, I encountered an API error. Please try again later."
    def search(self, pdf_path: str, question: str) -> str:
        """Query a PDF using a fine-tuned LLM without using RAG.
        
        This method extracts text from a PDF and directly queries it using the LLM.
        
        Args:
            pdf_path (str): Path to the PDF file.
            question (str): The question to ask the LLM.
        
        Returns:
            str: The LLM-generated answer.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        try:
            # Step 1: Parse PDF to extract raw text
            parser = LlamaParse(
                result_type="markdown",
                parse_table=True,
                table_output_format="markdown",
                num_workers=1
            )
            documents = parser.load_data(pdf_path)

            # Combine extracted text from all parsed documents
            full_text = "\n".join(doc.text for doc in documents)

            # Step 2: Ensure the text fits within LLM context limits
            MAX_TOKENS = 500  # Adjust based on model context window
            CHAR_PER_TOKEN = 4
            max_chars = MAX_TOKENS * CHAR_PER_TOKEN

            # If the text is too long, truncate it
            if len(full_text) > max_chars:
                full_text = full_text[:max_chars]

            # Step 3: Construct the LLM prompt
            prompt = (
                "<|begin_of_text|>"
                "<|start_header_id|>system<|end_header_id|>\n"
                "<|eot_id|>\n"
                "<|start_header_id|>user<|end_header_id|>\n"
                f"Document:\n{full_text}\n\n"
                f"Question: {question}<|eot_id|>\n"
                "<|start_header_id|>assistant<|end_header_id|>\n"
            )
            temp_llm = Groq(
                model=self._LLM_MODEL_NAME,
                api_key=os.getenv("GROQ_API_KEY"),
                temperature=1,  # Max allowed by most APIs
                max_tokens=1500,  # Longer responses
                top_p=0.95,       # Broader sampling
            )

            # Step 4: Get LLM response
            response = temp_llm.complete(prompt)
            return response.text.strip()

        except Exception as e:
            return f"Query failed: {str(e)}"
    def _format_prompt(self, context: str, question: str) -> str:
        """Generate structured prompt template for Llama 3.3"""
        return (
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>\n"
            "You are a technical research assistant. Use ONLY the context below to answer the question.\n"
            "Be concise, factual, and reference specific parts of the context where possible.\n"
            "If the answer isn't found in the context, simply say \"I don't know\".\n"
            "Provide complete and detailed answers ending with a fullstop.<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n"
        )
    def _validate_answer(self, response: str, truth: str) -> bool:
        """Simple validation using containment check"""
        return any(truth_word.lower() in response.lower() 
                 for truth_word in truth.split())
    def benchmark_models(self, pdf_path: str, test_questions: dict, models: list):
        """
        Compare model accuracy on a set of test questions
        
        Args:
            pdf_path: Path to test PDF document
            test_questions: Dictionary of {question: correct_answer}
            models: List of model names to compare
        """
        results = []
        
        for model_name in models:
            self.llm = Groq(
                model=model_name,
                api_key=os.getenv("GROQ_API_KEY"),
                temperature=0.1
            )
            self._reset_vector_store()
            self.process_pdf(pdf_path)
            correct = 0
            for question, truth in test_questions.items():
                response = self.query(question)
                if self._validate_answer(response, truth):
                    correct += 1
                    
            accuracy = (correct / len(test_questions)) * 100
            results.append({"Model": model_name, "Accuracy": accuracy})
            
        return pd.DataFrame(results)
def plot_accuracy(results_df):
    """Generate bar plot of model accuracies"""
    plt.figure(figsize=(10, 6))
    bars = plt.bar(results_df['Model'], results_df['Accuracy'], color='skyblue',width=0.5)
    plt.ylim(0, 110)
    plt.title('Model Accuracy Comparison', fontsize=14)
    plt.xlabel('LLM Models', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,  # x-position: center of the bar
            height + 1,                         # y-position: 1 point above the bar
            f'{height:.1f}%',                   # label text
            ha='center', va='bottom'
        )
    plt.tight_layout()
    plt.savefig('results.png')
    print("Plot saved as results.png")
    plt.show(block=True) 
def test_robustness(pdf_path: str, valid_questions: dict) -> dict:
        """Test system robustness and error handling capabilities"""
        test_results = {
            # Error Type: (success_count, total_tests)
            "missing_file": (0, 3),
            "malformed_pdf": (0, 3),
            "api_failure": (0, 3),
            "ambiguous_query": (0, 3)
        }
        
        rag = PDFRAGSystem()
        
        # Test 1: Missing file handling
        try:
            rag.process_pdf("non_existent_file.pdf")
        except FileNotFoundError:
            test_results["missing_file"] = (3, 3)
        except Exception as e:
            test_results["missing_file"] = (2, 3) if "not found" in str(e) else (1, 3)

        # Test 2: Malformed PDF handling
        try:
            with open("malformed.pdf", "w") as f:
                f.write("This is not a valid PDF")
            rag.process_pdf("malformed.pdf")
            test_results["malformed_pdf"] = (0, 3)
        except Exception as e:
            if "PDF processing failed" in str(e):
                test_results["malformed_pdf"] = (3, 3)
            else:
                test_results["malformed_pdf"] = (2, 3)
        finally:
            if os.path.exists("malformed.pdf"):
                os.remove("malformed.pdf")

        # Test 3: API failure simulation
        original_key = os.environ["GROQ_API_KEY"]
        try:
            os.environ["GROQ_API_KEY"] = "invalid_key"
            rag = PDFRAGSystem()
            test_results["api_failure"] = (0, 3)
        except Exception as e:
            print("API error encountered:", e)
            if "authentication" in str(e).lower():
                test_results["api_failure"] = (3, 3)
            else:
                test_results["api_failure"] = (2, 3)
        finally:
            os.environ["GROQ_API_KEY"] = original_key

        # Test 4: Ambiguous query handling
        rag.process_pdf(pdf_path)
        ambiguous_queries = [
            "What about the thing?",
            "Explain everything",
            "Tell me something important"
        ]
        success_count = 0
        for query in ambiguous_queries:
            try:
                response = rag.query(query).lower()
                if "don't know" in response or "context" in response:
                    success_count += 1
            except:
                pass
        test_results["ambiguous_query"] = (success_count, 3)

        return test_results    

def plot_error_coverage(results: dict):
    """Plot error handling coverage scores"""
    categories = list(results.keys())
    scores = [(v[0]/v[1])*100 for v in results.values()]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(categories, scores,width=0.5, color=['#4CAF50', '#2196F3', '#FF9800', '#9C27B0'])
    plt.ylim(0, 110)
    plt.title('Error Handling Coverage Scores', fontsize=14)
    plt.xlabel('Error Categories', fontsize=12)
    plt.ylabel('Handling Success Rate (%)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height+1,
                 f'{height:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('error_coverage.png')
    plt.show(block=True)

def plot_failure_modes(results: dict):
    """Plot chaos test failure rates"""
    categories = list(results.keys())
    failure_rates = [(1 - (v[0]/v[1]))*100 for v in results.values()]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(categories, failure_rates,width=0.5, color=['#F44336', '#E91E63', '#673AB7', '#009688'])
    plt.title('Chaos Test Failure Rates', fontsize=14)
    plt.xlabel('Failure Modes', fontsize=12)
    plt.ylabel('Failure Rate (%)', fontsize=12)
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('failure_rates.png')
    plt.show(block=True)
if __name__ == "__main__":
    try:
        import matplotlib
        matplotlib.use('TkAgg')  # Try 'Qt5Agg' if this doesn't work
        
        test_pdf = "Testdoc.pdf"
        questions = {
            "Explain the types of greenhouses from fig 1": "Natural greenhouse effect.Human enhanced greenhouse effect.",
            "Analyse fig 4 and answer which sector produces maximum annual greenhouse gas emission?": "Power stations",
            "The medical team from The Children's Hospital of Philadelphia examined the health proceedings of more than how many Americans alongside weather records? What did they discover?": "60,000 ,They discovered that individuals were most likely to be hospitalized with kidney stones three days after a temperature rise. "
        }
        
        models_to_test = [
            "llama-3.3-70b-versatile",
            "allam-2-7b",
            "mistral-saba-24b"
        ]
        
        rag = PDFRAGSystem()


    ######################################TESTING#######################################
        # print("Starting benchmark...")
        # rag.process_pdf(test_pdf)
        # results = rag.benchmark_models(test_pdf, questions, models_to_test)
        
        # print("\nBenchmark Results:")
        # print(results.to_string(index=False))
        
        # print("Generating plot...")
        # plot_accuracy(results)

        # print("\nRunning robustness tests...")
        # robustness_results = test_robustness(test_pdf, questions)
        
        # print("\nRobustness Test Results:")
        # for category, (success, total) in robustness_results.items():
        #     print(f"{category.replace('_', ' ').title():<20} {success}/{total} successful")
        
        # print("\nGenerating robustness graphs...")
        # plot_error_coverage(robustness_results)
        # plot_failure_modes(robustness_results)
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
    finally:
        input("Press Enter to exit...")  # Keep window open