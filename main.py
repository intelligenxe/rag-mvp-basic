"""
RAG Knowledge Base for NYSE Company
Using LlamaIndex with Groq LLM and Open Source Embeddings
"""
import os
from groq import Groq
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

from datetime import datetime
from typing import List, Dict

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Document,
    Settings
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# LlamaHub data loaders
from llama_index.readers.web import BeautifulSoupWebReader, TrafilaturaWebReader
from llama_index.readers.file import PDFReader, UnstructuredReader


class NYSECompanyRAG:
    """RAG system for NYSE company public information"""
    
    def __init__(self, company_ticker: str, company_name: str, groq_api_key: str = None):
        self.ticker = company_ticker
        self.company_name = company_name
        self.documents = []
        
        ### LLM API
        # Get Groq API key
        if groq_api_key:
            os.environ["GROQ_API_KEY"] = groq_api_key
        elif not os.environ.get("GROQ_API_KEY"):
            raise ValueError("Groq API key must be provided or set in GROQ_API_KEY environment variable")
        
        ### LLM
        # Configure LlamaIndex settings with Groq and open source embeddings
        Settings.llm = Groq(
            model="llama-3.3-70b-versatile",  # Options: mixtral-8x7b-32768, llama-3.3-70b-versatile, etc.
            temperature=0.1,
            api_key=os.environ.get("GROQ_API_KEY")
        )
        
        ### Embeddings Model
        # Use HuggingFace open source embedding model
        # Options: 
        # - "BAAI/bge-small-en-v1.5" (lightweight, 384 dims)
        # - "BAAI/bge-base-en-v1.5" (balanced, 768 dims)
        # - "sentence-transformers/all-MiniLM-L6-v2" (popular, 384 dims)
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5"
        )
        
        ### Chunking 
        Settings.node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=200)

        ### Vector Store
        # Initialize vector store
        self.chroma_client = chromadb.PersistentClient(path=f"./chroma_db_{company_ticker}")
        self.chroma_collection = self.chroma_client.get_or_create_collection(
            name=f"{company_ticker}_knowledge_base"
        )
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        
        ### Index & Query Engine
        self.index = None
        self.query_engine = None
    
    # ===== DATA COLLECTION =====

    ### Load SEC Filings
    def load_sec_filings(self, filing_types: List[str] = ["10-K", "10-Q", "8-K"]):
        """Load SEC filings from EDGAR"""
        print(f"Loading SEC filings for {self.ticker}...")
        
        # Option 1: Use SECFilingsLoader (if available)
        # loader = SECFilingsLoader(ticker=self.ticker, filing_types=filing_types)
        # documents = loader.load_data()
        
        # Option 2: Manual download + parse
        # You can download from: https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}
        # Then use UnstructuredReader or PDFReader
        
        sec_docs = []
        # TODO: Implement actual SEC filing loading
        # For now, placeholder:
        for filing_type in filing_types:
            doc = Document(
                text=f"Placeholder for {filing_type} filing",
                metadata={
                    "source": "SEC",
                    "filing_type": filing_type,
                    "ticker": self.ticker,
                    "date": datetime.now().isoformat()
                }
            )
            sec_docs.append(doc)
        
        self.documents.extend(sec_docs)
        print(f"Loaded {len(sec_docs)} SEC filing documents")
        return sec_docs
    
    ### Load PDFs
    def load_annual_reports(self, pdf_paths: List[str]):
        """Load annual reports from PDF files"""
        print("Loading annual reports...")
        pdf_reader = PDFReader()
        
        annual_docs = []
        for pdf_path in pdf_paths:
            docs = pdf_reader.load_data(file=pdf_path)
            
            # Add metadata
            for doc in docs:
                doc.metadata.update({
                    "source": "Annual Report",
                    "ticker": self.ticker,
                    "file_path": pdf_path
                })
            
            annual_docs.extend(docs)
        
        self.documents.extend(annual_docs)
        print(f"Loaded {len(annual_docs)} annual report documents")
        return annual_docs

    ### Load Website
    def load_company_website(self, urls: List[str]):
        """Load content from company website"""
        print("Loading company website content...")
        web_reader = BeautifulSoupWebReader()
        
        web_docs = []
        for url in urls:
            try:
                docs = web_reader.load_data(urls=[url])
                
                # Add metadata
                for doc in docs:
                    doc.metadata.update({
                        "source": "Company Website",
                        "ticker": self.ticker,
                        "url": url,
                        "scrape_date": datetime.now().isoformat()
                    })
                
                web_docs.extend(docs)
            except Exception as e:
                print(f"Error loading {url}: {e}")
        
        self.documents.extend(web_docs)
        print(f"Loaded {len(web_docs)} website documents")
        return web_docs
    
    ### Load News  
    def load_news_releases(self, rss_url: str = None, news_urls: List[str] = None):
        """Load news releases and earnings announcements"""
        print("Loading news releases...")
        
        news_docs = []
        
        # Option 1: RSS feed
        if rss_url:
            # from llama_index.readers.rss import RssReader
            # rss_reader = RssReader()
            # docs = rss_reader.load_data(urls=[rss_url])
            pass
        
        # Option 2: Direct URLs
        if news_urls:
            web_reader = TrafilaturaWebReader()
            for url in news_urls:
                try:
                    docs = web_reader.load_data(urls=[url])
                    for doc in docs:
                        doc.metadata.update({
                            "source": "News Release",
                            "ticker": self.ticker,
                            "url": url,
                            "scrape_date": datetime.now().isoformat()
                        })
                    news_docs.extend(docs)
                except Exception as e:
                    print(f"Error loading news from {url}: {e}")
        
        self.documents.extend(news_docs)
        print(f"Loaded {len(news_docs)} news documents")
        return news_docs
    
    # ===== INDEX CONSTRUCTION =====
    
    def build_index(self):
        """Build vector index from all loaded documents"""
        print(f"\nBuilding index from {len(self.documents)} documents...")
        
        if not self.documents:
            raise ValueError("No documents loaded. Load data first.")
        
        # Create index
        self.index = VectorStoreIndex.from_documents(
            self.documents,
            storage_context=self.storage_context,
            show_progress=True
        )
        
        print("Index built successfully!")
        return self.index
    
    def load_existing_index(self):
        """Load previously built index"""
        print("Loading existing index...")
        self.index = VectorStoreIndex.from_vector_store(
            self.vector_store,
            storage_context=self.storage_context
        )
        print("Index loaded successfully!")
        return self.index
    
    # ===== QUERY & RETRIEVAL =====
    
    def create_query_engine(self, similarity_top_k: int = 5):
        """Create query engine with retrieval configuration"""
        if not self.index:
            raise ValueError("Index not built. Call build_index() first.")
        
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=similarity_top_k,
            response_mode="compact",  # Options: compact, refine, tree_summarize
            verbose=True
        )
        
        return self.query_engine
    
    def query(self, question: str) -> str:
        """Query the knowledge base"""
        if not self.query_engine:
            self.create_query_engine()
        
        print(f"\nQuery: {question}")
        response = self.query_engine.query(question)
        
        # Print sources
        print("\nSources:")
        for node in response.source_nodes:
            print(f"- {node.metadata.get('source', 'Unknown')}: {node.metadata.get('file_path', node.metadata.get('url', 'N/A'))}")
        
        return response
    
    def query_with_filters(self, question: str, source_filter: str = None, 
                          date_range: tuple = None):
        """Query with metadata filters"""
        from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
        
        filters = []
        if source_filter:
            filters.append(ExactMatchFilter(key="source", value=source_filter))
        
        # Add more filters as needed
        metadata_filters = MetadataFilters(filters=filters) if filters else None
        
        query_engine = self.index.as_query_engine(
            similarity_top_k=5,
            filters=metadata_filters
        )
        
        response = query_engine.query(question)
        return response
    
    # ===== MAINTENANCE =====
    
    def update_with_new_data(self, new_documents: List[Document]):
        """Add new documents to existing index"""
        print(f"Adding {len(new_documents)} new documents to index...")
        
        for doc in new_documents:
            self.index.insert(doc)
        
        print("Index updated!")
    
    def get_stats(self) -> Dict:
        """Get statistics about the knowledge base"""
        doc_sources = {}
        for doc in self.documents:
            source = doc.metadata.get("source", "Unknown")
            doc_sources[source] = doc_sources.get(source, 0) + 1
        
        return {
            "total_documents": len(self.documents),
            "documents_by_source": doc_sources,
            "ticker": self.ticker,
            "company_name": self.company_name
        }


# ===== EXAMPLE USAGE =====

def main():
    # Initialize RAG system with Groq API key
    rag = NYSECompanyRAG(
        company_ticker="AAPL",
        company_name="Apple Inc.",
        groq_api_key=os.environ.get("GROQ_API_KEY")
    )
    
    # Load data from various sources
    # rag.load_sec_filings(filing_types=["10-K", "10-Q"])
    rag.load_annual_reports(pdf_paths=["./data/apple_annual_report_2024.pdf"])
    # rag.load_company_website(urls=[
    #     "https://www.apple.com/investor-relations/",
    #     "https://www.apple.com/newsroom/"
    # ])
    # rag.load_news_releases(news_urls=[
    #     "https://www.apple.com/newsroom/2024/01/apple-reports-first-quarter-results/"
    # ])
    
    # Build index
    rag.build_index()
    
    # Or load existing index
    # rag.load_existing_index()
    
    # Create query engine
    rag.create_query_engine()
    
    # Query examples
    response = rag.query("What was the revenue in the last fiscal year?")
    print(f"\nAnswer: {response}")
    
    response = rag.query("What are the main business segments?")
    print(f"\nAnswer: {response}")
    
    # Query with filters
    # response = rag.query_with_filters(
    #     "What were the key highlights?",
    #     source_filter="Annual Report"
    # )
    
    # Get statistics
    # stats = rag.get_stats()
    # print(f"\nKnowledge Base Stats: {stats}")
    
    print("RAG system initialized with Groq LLM and open source embeddings.")
    print("Uncomment code sections to use.")


if __name__ == "__main__":
    main()