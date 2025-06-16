# memory/manager.py

import os
import json
import threading
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import deque
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class MemoryManager:
    def __init__(
        self,
        episodic_path: str = "memory/episodic.json",
        semantic_path: str = "memory/semantic.json",
        max_episodic_in_memory: int = 100,
        duplicate_threshold: float = 0.85
    ):
        """
        Enhanced memory manager with in-memory caching, deduplication,
        and semantic search capabilities.
        """
        self.episodic_path = episodic_path
        self.semantic_path = semantic_path
        self.max_episodic_in_memory = max_episodic_in_memory
        self.duplicate_threshold = duplicate_threshold
        self.lock = threading.Lock()  # Thread safety
        
        # Create directories if needed
        os.makedirs(os.path.dirname(episodic_path), exist_ok=True)
        os.makedirs(os.path.dirname(semantic_path), exist_ok=True)
        
        # Initialize in-memory caches
        self.episodic_memory = deque(maxlen=max_episodic_in_memory)
        self.semantic_memory = []
        
        # Load existing data
        self._load_initial_data()
        
        # Initialize semantic search
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self._update_semantic_index()

    def _load_initial_data(self):
        """Load existing data from disk with error handling"""
        try:
            if os.path.exists(self.episodic_path):
                with open(self.episodic_path, 'r') as f:
                    self.episodic_memory = deque(
                        json.load(f)[-self.max_episodic_in_memory:],
                        maxlen=self.max_episodic_in_memory
                    )
        except (FileNotFoundError, json.JSONDecodeError):
            # Initialize with empty file
            with open(self.episodic_path, 'w') as f:
                json.dump([], f)
                
        try:
            if os.path.exists(self.semantic_path):
                with open(self.semantic_path, 'r') as f:
                    self.semantic_memory = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            with open(self.semantic_path, 'w') as f:
                json.dump([], f)
    
    def _update_semantic_index(self):
        """Update TF-IDF vectors for semantic search"""
        if not self.semantic_memory:
            return
            
        texts = [item['summary'] for item in self.semantic_memory]
        self.semantic_vectors = self.vectorizer.fit_transform(texts)
    
    def _is_duplicate_episodic(self, thought: str) -> bool:
        """Check if thought is similar to recent memories"""
        if not self.episodic_memory:
            return False
            
        # Compare against recent memories
        recent_texts = [e['thought'] for e in self.episodic_memory]
        recent_texts.append(thought)
        vectors = self.vectorizer.transform(recent_texts)
        
        # Calculate similarity to most recent memories
        similarities = cosine_similarity(vectors[-1:], vectors[:-1])
        return np.max(similarities) > self.duplicate_threshold

    def add_episodic(
        self,
        thought: str,
        source: str = "subconscious",
        salience: float = 0.5
    ) -> None:
        """
        Add a new episodic memory with deduplication and automatic persistence
        """
        # Skip duplicates
        if self._is_duplicate_episodic(thought):
            return
            
        with self.lock:
            entry = {
                "timestamp": datetime.now().isoformat(timespec='seconds'),
                "thought": thought,
                "source": source,
                "salience": salience
            }
            
            # Update in-memory cache
            self.episodic_memory.append(entry)
            
            # Persist to disk in background
            threading.Thread(target=self._persist_episodic, daemon=True).start()

    def _persist_episodic(self):
        """Persist episodic memory to disk (run in background thread)"""
        with self.lock:
            try:
                # Read existing data and append
                if os.path.exists(self.episodic_path):
                    with open(self.episodic_path, 'r') as f:
                        existing = json.load(f)
                else:
                    existing = []
                
                # Append new memories
                existing.extend(self.episodic_memory)
                
                # Write back
                with open(self.episodic_path, 'w') as f:
                    json.dump(existing[-self.max_episodic_in_memory*2:], f, indent=2)
            except Exception as e:
                print(f"Episodic persistence error: {e}")

    def get_recent_memories(self, n: int = 5) -> List[str]:
        """Get most recent memory texts"""
        with self.lock:
            return [e['thought'] for e in list(self.episodic_memory)[-n:]]
    
    def get_recent_episodic(self, n: int = 5) -> List[Dict]:
        """Get full recent memory entries"""
        with self.lock:
            return list(self.episodic_memory)[-n:]
    
    def get_relevant_semantic(self, query: str, n: int = 3) -> List[str]:
        """Get semantically relevant memories using TF-IDF cosine similarity"""
        with self.lock:
            if not self.semantic_memory:
                return []
            
            # Vectorize query
            query_vec = self.vectorizer.transform([query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_vec, self.semantic_vectors)
            indices = np.argsort(similarities[0])[-n:][::-1]
            
            return [self.semantic_memory[i]['summary'] for i in indices]

    def consolidate_semantic(self, salience_threshold: float = 0.7) -> None:
        """
        Consolidate high-salience memories into semantic storage with summarization
        """
        with self.lock:
            # Get high-salience memories
            high_salience = [
                e for e in self.episodic_memory
                if e.get("salience", 0) >= salience_threshold
            ]
            
            # Skip if no new high-salience memories
            if not high_salience:
                return
                
            # Group similar memories (simple implementation)
            new_semantic = []
            for memory in high_salience:
                # Check if similar to existing semantic memory
                similar_exists = any(
                    self._text_similarity(memory['thought'], s['summary']) > 0.8
                    for s in self.semantic_memory
                )
                
                if not similar_exists:
                    new_semantic.append({
                        "timestamp": datetime.now().isoformat(timespec='seconds'),
                        "summary": memory['thought'],  # Placeholder for actual summarization
                        "source": memory.get("source", "subconscious"),
                        "origin_timestamp": memory['timestamp']
                    })
            
            # Update semantic memory
            self.semantic_memory.extend(new_semantic)
            
            # Persist and update index
            self._persist_semantic()
            self._update_semantic_index()
    
    def _text_similarity(self, a: str, b: str) -> float:
        """Calculate text similarity score"""
        vecs = self.vectorizer.transform([a, b])
        return cosine_similarity(vecs[0], vecs[1])[0][0]
    
    def _persist_semantic(self):
        """Persist semantic memory to disk"""
        try:
            with open(self.semantic_path, 'w') as f:
                json.dump(self.semantic_memory, f, indent=2)
        except Exception as e:
            print(f"Semantic persistence error: {e}")
    
    def add_conscious_memory(self, evaluation: Dict) -> None:
        """
        Add a conscious-processed memory with enhanced metadata
        """
        self.add_episodic(
            thought=evaluation['refined'],
            source="conscious",
            salience=evaluation['salience']
        )
        
        # Auto-consolidate high-value thoughts
        if evaluation['salience'] >= 0.7:
            with self.lock:
                self.semantic_memory.append({
                    "timestamp": datetime.now().isoformat(timespec='seconds'),
                    "summary": evaluation['refined'],
                    "category": evaluation.get('category', 'misc'),
                    "origin": "conscious",
                    "salience": evaluation['salience']
                })
                self._persist_semantic()
                self._update_semantic_index()