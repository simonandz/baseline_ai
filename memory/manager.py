import sqlite3
import numpy as np
import logging
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from transformers import pipeline

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class MemoryManager:
    def __init__(self, db_path: str = "memory.db", embedder_model: str = "all-MiniLM-L6-v2"):
        """
        Manages AI memory using SQLite database with semantic search capabilities
        
        Features:
        - Automatic database initialization
        - Embedding-based memory storage
        - Context-aware retrieval
        - Periodic memory consolidation
        - Salience scoring
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._init_database()
        
        # Load models
        logger.info("Loading embedding model: %s", embedder_model)
        self.embedder = SentenceTransformer(embedder_model)
        
        # Load summarization model on demand
        self.summarizer = None
        self.summarizer_model = "facebook/bart-large-cnn"
        
        # Cache for recent memories
        self.recent_memories = []
        self.last_consolidation = datetime.now()
        
        logger.info("MemoryManager initialized")

    def _init_database(self):
        """Create database schema if not exists"""
        cursor = self.conn.cursor()
        
        # Thoughts table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS thoughts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            embedding BLOB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            salient_score FLOAT DEFAULT 0,
            is_processed BOOLEAN DEFAULT 0
        )
        ''')
        
        # Memory consolidations (summaries)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS memory_consolidations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            summary TEXT NOT NULL,
            key_themes TEXT,  -- JSON array of themes
            thought_ids TEXT,  -- JSON array of IDs
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Feedback system
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            thought_id INTEGER NOT NULL,
            feedback_type TEXT CHECK(feedback_type IN ('positive', 'negative')),
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (thought_id) REFERENCES thoughts (id)
        )
        ''')
        
        self.conn.commit()
        logger.info("Database schema initialized")

    def add_memory(self, content: str, salient_score: float = 0.0):
        """
        Store a new memory with semantic embedding
        
        Args:
            content: The text of the memory
            salient_score: Importance rating (0.0-1.0)
        """
        try:
            embedding = self.embedder.encode([content])[0]
            cursor = self.conn.cursor()
            
            cursor.execute('''
                INSERT INTO thoughts (content, embedding, salient_score)
                VALUES (?, ?, ?)
            ''', (content, embedding.tobytes(), salient_score))
            
            self.conn.commit()
            
            # Cache recent memory
            self.recent_memories.insert(0, content)
            if len(self.recent_memories) > 20:
                self.recent_memories.pop()
                
            logger.debug("Added memory: %s", content[:50] + "...")
            return cursor.lastrowid
        except Exception as e:
            logger.error("Failed to add memory: %s", e)
            return None

    def get_recent_memories(self, count: int = 5) -> List[str]:
        """Get most recent memories from cache"""
        return self.recent_memories[:count]

    def semantic_search(self, query: str, count: int = 3) -> List[Dict[str, Any]]:
        """
        Find semantically similar memories
        
        Args:
            query: Text to search for
            count: Number of results to return
            
        Returns:
            List of {id, content, similarity} dictionaries
        """
        try:
            query_embedding = self.embedder.encode([query])[0]
            cursor = self.conn.cursor()
            
            # Get all embeddings
            cursor.execute("SELECT id, content, embedding FROM thoughts")
            memories = cursor.fetchall()
            
            # Calculate similarities
            results = []
            for mem_id, content, emb_blob in memories:
                memory_embedding = np.frombuffer(emb_blob, dtype=np.float32)
                similarity = cosine_similarity(
                    [query_embedding], 
                    [memory_embedding]
                )[0][0]
                results.append({
                    "id": mem_id,
                    "content": content,
                    "similarity": similarity
                })
            
            # Return top matches
            return sorted(results, key=lambda x: x["similarity"], reverse=True)[:count]
        except Exception as e:
            logger.error("Semantic search failed: %s", e)
            return []

    def consolidate_memory(self, force: bool = False):
        """
        Summarize and cluster recent memories
        
        Args:
            force: Run consolidation regardless of schedule
        """
        # Only consolidate once per hour
        if not force and datetime.now() - self.last_consolidation < timedelta(hours=1):
            return
            
        logger.info("Starting memory consolidation...")
        try:
            cursor = self.conn.cursor()
            
            # Get unprocessed memories from last 24 hours
            cursor.execute('''
                SELECT id, content 
                FROM thoughts 
                WHERE created_at > datetime('now', '-1 day')
                AND is_processed = 0
            ''')
            recent_memories = cursor.fetchall()
            
            if not recent_memories:
                logger.info("No new memories to consolidate")
                return
                
            # Extract content for clustering
            contents = [mem[1] for mem in recent_memories]
            memory_ids = [mem[0] for mem in recent_memories]
            
            # Cluster memories
            embeddings = self.embedder.encode(contents)
            n_clusters = min(5, len(contents) // 3)  # Adaptive clustering
            if n_clusters < 1:
                n_clusters = 1
                
            clusters = KMeans(n_clusters=n_clusters).fit(embeddings)
            
            # Summarize each cluster
            if self.summarizer is None:
                logger.info("Loading summarization model: %s", self.summarizer_model)
                self.summarizer = pipeline(
                    "summarization", 
                    model=self.summarizer_model,
                    truncation=True
                )
                
            summaries = []
            for cluster_id in range(n_clusters):
                # Get cluster contents
                cluster_contents = [
                    contents[i] for i in range(len(contents)) 
                    if clusters.labels_[i] == cluster_id
                ]
                
                # Generate summary
                summary = self.summarizer(
                    "\n".join(cluster_contents),
                    max_length=150,
                    min_length=30,
                    do_sample=False
                )[0]['summary_text']
                
                # Store consolidation
                cluster_ids = [
                    memory_ids[i] for i in range(len(memory_ids))
                    if clusters.labels_[i] == cluster_id
                ]
                
                cursor.execute('''
                    INSERT INTO memory_consolidations 
                    (summary, thought_ids, key_themes) 
                    VALUES (?, ?, ?)
                ''', (
                    summary,
                    json.dumps(cluster_ids),
                    json.dumps([f"Theme {cluster_id+1}"])  # Simplified
                ))
                
                summaries.append(summary)
            
            # Mark memories as processed
            cursor.execute('''
                UPDATE thoughts 
                SET is_processed = 1 
                WHERE id IN ({})
            '''.format(",".join(map(str, memory_ids))))
            
            self.conn.commit()
            self.last_consolidation = datetime.now()
            logger.info("Memory consolidation complete. Created %d summaries", n_clusters)
            return summaries
        except Exception as e:
            logger.error("Memory consolidation failed: %s", e)
            return []

    def add_feedback(self, thought_id: int, feedback_type: str, notes: str = ""):
        """Record user feedback on a memory"""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO feedback (thought_id, feedback_type, notes)
                VALUES (?, ?, ?)
            ''', (thought_id, feedback_type, notes))
            
            # Update salience score based on feedback
            adjustment = 0.2 if feedback_type == "positive" else -0.1
            cursor.execute('''
                UPDATE thoughts
                SET salient_score = salient_score + ?
                WHERE id = ?
            ''', (adjustment, thought_id))
            
            self.conn.commit()
            logger.info("Recorded %s feedback for thought %d", feedback_type, thought_id)
            return True
        except Exception as e:
            logger.error("Failed to record feedback: %s", e)
            return False

    def get_consolidations(self, days: int = 7) -> List[Dict[str, Any]]:
        """Retrieve recent memory consolidations"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT id, summary, key_themes, created_at
            FROM memory_consolidations
            WHERE created_at > datetime('now', ?)
            ORDER BY created_at DESC
        ''', (f"-{days} days",))
        
        return [{
            "id": row[0],
            "summary": row[1],
            "themes": json.loads(row[2]),
            "created_at": row[3]
        } for row in cursor.fetchall()]

    def close(self):
        """Clean up resources"""
        self.conn.close()
        logger.info("MemoryManager shutdown")

    def __del__(self):
        self.close()