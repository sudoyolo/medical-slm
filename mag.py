#!/usr/bin/env python3
"""
Multi-Agent Qwen Framework
Runs 3 agents asynchronously: Simple, Web Search, and Medical RAG
"""

import asyncio
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from pinecone import Pinecone, ServerlessSpec
import os
from datetime import datetime
from typing import Dict, List
import ddgs



class BaseAgent:
    """Base class for all agents"""
    def __init__(self, name: str, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
        self.name = name
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
    
    async def load_model(self):
        """Load the Qwen model"""
        print(f"[{self.name}] Loading model...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        print(f"[{self.name}] Model loaded!")
    
    def generate_response(self, prompt: str, max_tokens: int = 512) -> str:
        """Generate response from model"""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
    
    async def process(self, query: str) -> Dict:
        """Process query - to be overridden by subclasses"""
        raise NotImplementedError


class FineTunedAgent(BaseAgent):
    """Fine-tuned merged model agent"""
    def __init__(self, model_path: str = "/home/ubuntu/environment/ml/qwen/merged_model"):
        super().__init__("Fine-Tuned Medical Agent", model_path)
    
    def generate_response(self, query: str, max_tokens: int = 2048) -> str:
        """Generate response using fine-tuned model"""
        prompt = f"""<|im_start|>system
You are medical diagnoser. 
The patient will talk to you about their medical condition or illnesses of symptoms they might have.
You have to advise them appropriately. 
Mention any relevant diseases, health checks and medicines that the patient should look into.<|im_end|>
<|im_start|>user
{query}<|im_end|>
<|im_start|>assistant
"""
        model_inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
        
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    async def process(self, query: str) -> Dict:
        """Process query with fine-tuned model"""
        start_time = datetime.now()
        print(f"[{self.name}] Processing query...")
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, self.generate_response, query)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        return {
            "agent": self.name,
            "response": response,
            "elapsed_time": elapsed,
            "context": "Fine-tuned merged model"
        }


class WebSearchAgent(BaseAgent):
    """Qwen agent with web search capability"""
    def __init__(self):
        super().__init__("Web Search Agent")
    
    def web_search(self, query: str, max_results: int = 3) -> str:
        """Perform web search using DuckDuckGo"""
        try:
            # with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            
            search_context = ""
            for i, result in enumerate(results, 1):
                search_context += f"\n[Result {i}]\n"
                search_context += f"Title: {result['title']}\n"
                search_context += f"Content: {result['body']}\n"
            
            return search_context
        except Exception as e:
            return f"Search error: {str(e)}"
    
    async def process(self, query: str) -> Dict:
        """Process query with web search"""
        start_time = datetime.now()
        print(f"[{self.name}] Searching web...")
        
        # Run search in executor
        loop = asyncio.get_event_loop()
        search_results = await loop.run_in_executor(
            None,
            self.web_search,
            query
        )
        
        # Create prompt with search results
        prompt = f"""You are a helpful assistant with access to web search results.

User Question: {query}

Web Search Results:
{search_results}

Based on the search results above, provide a comprehensive and accurate answer."""
        
        print(f"[{self.name}] Generating response...")
        response = await loop.run_in_executor(
            None,
            self.generate_response,
            prompt
        )
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        return {
            "agent": self.name,
            "response": response,
            "elapsed_time": elapsed,
            "context": "Web search + Model"
        }


class MedicalRAGAgent(BaseAgent):
    """Qwen agent with medical knowledge base (RAG) using Pinecone"""
    def __init__(self, pinecone_api_key=None):
        super().__init__("Medical RAG Agent")
        self.embedder = None
        self.index = None
        self.pc = None
        self.index_name = "medical-terms"
        self.pinecone_api_key = "pcsk_WCHvg_RWLe96SyvQSpxCsefR3uhrWffdQNmPCeE6w55BQW5BVPbahFvTaSqY3f2JgCr86"
        
        if not self.pinecone_api_key:
            print(f"[{self.name}] WARNING: No Pinecone API key found!")
            print(f"[{self.name}] Set PINECONE_API_KEY environment variable or pass it during initialization")
    
    async def load_model(self):
        """Load Qwen model, embeddings, and connect to Pinecone"""
        await super().load_model()
        
        print(f"[{self.name}] Loading embedding model...")
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        if not self.pinecone_api_key:
            print(f"[{self.name}] ERROR: Cannot connect to Pinecone without API key!")
            return
        
        try:
            print(f"[{self.name}] Connecting to Pinecone...")
            self.pc = Pinecone(api_key=self.pinecone_api_key)
            
            # Check if index exists
            existing_indexes = self.pc.list_indexes().names()
            
            if self.index_name not in existing_indexes:
                print(f"[{self.name}] Creating Pinecone index '{self.index_name}'...")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=384,  # all-MiniLM-L6-v2 dimension
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'
                    )
                )
                print(f"[{self.name}] Index created!")
            
            # Connect to index
            self.index = self.pc.Index(self.index_name)
            
            # Check if index needs to be populated
            stats = self.index.describe_index_stats()
            if stats['total_vector_count'] == 0:
                print(f"[{self.name}] Index is empty. Populating with medical terms...")
                await self.create_index()
            else:
                print(f"[{self.name}] Connected! Index has {stats['total_vector_count']} vectors")
        
        except Exception as e:
            print(f"[{self.name}] ERROR connecting to Pinecone: {e}")
            self.index = None
    
    async def create_index(self):
        """Populate Pinecone index from medical dataset"""
        print(f"[{self.name}] Downloading wiki_medical_terms dataset...")
        dataset = load_dataset("gamino/wiki_medical_terms", split="train")
        
        batch_size = 100
        vectors = []
        
        print(f"[{self.name}] Processing {len(dataset)} medical terms...")
        for i, item in enumerate(dataset):
            if i % 1000 == 0:
                print(f"[{self.name}] Processed {i}/{len(dataset)} terms...")
            
            term = item.get('page_title', '')
            definition = item.get('page_text', '')
            combined_text = f"{term}: {definition}"
            
            # Create embedding
            embedding = self.embedder.encode(combined_text).tolist()
            
            # Prepare vector for Pinecone
            vectors.append({
                'id': f"term_{i}",
                'values': embedding,
                'metadata': {
                    'term': term,
                    'definition': definition[:1000]  # Pinecone metadata size limits
                }
            })
            
            # Upload in batches
            if len(vectors) >= batch_size:
                self.index.upsert(vectors=vectors)
                vectors = []
        
        # Upload remaining vectors
        if vectors:
            self.index.upsert(vectors=vectors)
        
        print(f"[{self.name}] Pinecone index populated with medical terms!")
    
    def retrieve_context(self, query: str) -> str:
        """Retrieve relevant medical context from Pinecone"""
        if self.index is None:
            return "Medical knowledge base not available. Please check Pinecone connection."
        
        try:
            # Encode query
            query_embedding = self.embedder.encode(query).tolist()
            
            # Search in Pinecone
            results = self.index.query(
                vector=query_embedding,
                include_metadata=True
            )
            
            # Format context
            context = ""
            for i, match in enumerate(results['matches']):
                metadata = match['metadata']
                context += f"\n[Medical Term {i+1}] (Similarity: {match['score']:.3f})\n"
                context += f"Term: {metadata['term']}\n"
                context += f"Definition: {metadata['definition']}\n"
            
            return context
        except Exception as e:
            return f"Error retrieving context: {str(e)}"
    
    async def process(self, query: str) -> Dict:
        """Process query with medical RAG"""
        start_time = datetime.now()
        print(f"[{self.name}] Retrieving medical knowledge...")
        
        # Run retrieval in executor
        loop = asyncio.get_event_loop()
        medical_context = await loop.run_in_executor(
            None,
            self.retrieve_context,
            query
        )
        
        # Create prompt with medical knowledge
        prompt = f"""You are a medical assistant with access to a comprehensive medical knowledge base.

User Question: {query}

Relevant Medical Knowledge:
{medical_context}

Based on the medical knowledge provided above, answer the question accurately and clearly.

Important: This is for educational purposes only and should not replace professional medical advice."""
        
        print(f"[{self.name}] Generating response...")
        response = await loop.run_in_executor(
            None,
            self.generate_response,
            prompt
        )
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        return {
            "agent": self.name,
            "response": response,
            "elapsed_time": elapsed,
            "context": "Medical RAG + Model"
        }


class MultiAgentOrchestrator:
    """Orchestrates multiple agents running in parallel"""
    def __init__(self):
        self.agents: List[BaseAgent] = []
        self.initialized = False
    
    async def initialize(self):
        """Initialize all agents"""
        print("="*80)
        print("Initializing Multi-Agent System")
        print("="*80)
        
        # Get Pinecone API key for Medical RAG Agent
        pinecone_api_key = "pcsk_WCHvg_RWLe96SyvQSpxCsefR3uhrWffdQNmPCeE6w55BQW5BVPbahFvTaSqY3f2JgCr86"
        if not pinecone_api_key:
            print("\n⚠️  WARNING: PINECONE_API_KEY not found in environment")
            print("Medical RAG Agent will not be able to connect to Pinecone.")
            print("Set it with: export PINECONE_API_KEY='your-api-key'\n")
        
        # Create agents
        self.agents = [
            FineTunedAgent(),
            WebSearchAgent(),
            MedicalRAGAgent(pinecone_api_key=pinecone_api_key)
        ]
        
        # Load models for all agents in parallel
        await asyncio.gather(*[agent.load_model() for agent in self.agents])
        
        self.initialized = True
        print("\n" + "="*80)
        print("All agents ready!")
        print("="*80 + "\n")
    
    async def query_all_agents(self, user_query: str) -> List[Dict]:
        """Query all agents in parallel and wait for all responses"""
        if not self.initialized:
            await self.initialize()
        
        print(f"\n{'='*80}")
        print(f"QUERY: {user_query}")
        print(f"{'='*80}")
        print("Running all agents in parallel...\n")
        
        # Run all agents asynchronously
        tasks = [agent.process(user_query) for agent in self.agents]
        
        # Use asyncio.gather as barrier - waits for all to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append({
                    "agent": "Unknown",
                    "response": f"Error: {str(result)}",
                    "elapsed_time": 0,
                    "context": "Error occurred"
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    def display_results(self, results: List[Dict]):
        """Display results from all agents"""
        print("\n" + "="*80)
        print("RESPONSES FROM ALL AGENTS")
        print("="*80 + "\n")
        
        for i, result in enumerate(results, 1):
            print(f"{'─'*80}")
            print(f"Agent {i}: {result['agent']}")
            print(f"Context: {result['context']}")
            print(f"Time: {result['elapsed_time']:.2f}s")
            print(f"{'─'*80}")
            print(f"\n{result['response']}\n")
        
        print("="*80 + "\n")


async def main():
    """Main application loop"""
    orchestrator = MultiAgentOrchestrator()
    
    # Initialize all agents
    await orchestrator.initialize()
    
    print("Multi-Agent Qwen System Ready!")
    print("Type 'quit' or 'exit' to stop\n")
    
    while True:
        try:
            user_input = input("Your question: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit']:
                print("\nGoodbye!")
                break
            
            # Query all agents in parallel
            results = await orchestrator.query_all_agents(user_input)
            
            # Display all results
            orchestrator.display_results(results)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    # Install requirements:
    # pip install torch transformers duckduckgo-search sentence-transformers datasets faiss-cpu accelerate
    asyncio.run(main())