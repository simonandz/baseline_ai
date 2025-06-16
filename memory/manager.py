import os
import sqlite3
import numpy as np
import faiss
from datetime import datetime
from sentence_transformers import SentenceTransformer
import logging
from typing import List, Optional
import threading

logger = logging.getLogger(__name__)

class MemoryManager:
    def __init__(self, db_path: str = "memory/mind.db"):
        """
        Enhanced memory manager with:
        - SQLite storage
        - FAISS semantic search
        - Thread-safe operations
        
        Args:
            db_path: Path to SQLite database file
        """
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.lock = threading.Lock()
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize database
        self._init_db()
        logger.info(f"MemoryManager initialized with database at {db_path}")

    def _init_db(self):
        """Initialize database tables"""
        with self.lock:
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS thoughts (
                id INTEGER PRIMARY KEY,
                timestamp TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding BLOB,
                salience REAL DEFAULT 0.5,
                category TEXT
            )""")
            self.conn.commit()

    def add_thought(
        self,
        content: str,
        salience: float = 0.5,
        category: Optional[str] = None
    ) -> bool:
        """
        Add a thought to memory with semantic embedding
        
        Args:
            content: The thought text
            salience: Importance score (0.0-1.0)
            category: Optional category label
            
        Returns:
            bool: True if successfully added
        """
        try:
            embedding = self.embedder.encode([content])[0]
            
            with self.lock:
                self.conn.execute(
                    """INSERT INTO thoughts 
                    (timestamp, content, embedding, salience, category)
                    VALUES (?, ?, ?, ?, ?)""",
                    (
                        datetime.now().isoformat(),
                        content,
                        embedding.tobytes(),
                        salience,
                        category
                    )
                )
                self.conn.commit()
            
            logger.debug(f"Added thought to memory: {content[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add thought: {e}")
            return False

    def get_recent_memories(self, limit: int = 5) -> List[str]:
        """
        Get most recent thoughts
        
        Args:
            limit: Maximum number of thoughts to return
            
        Returns:
            List of thought contents
        """
        try:
            with self.lock:
                cursor = self.conn.cursor()
                cursor.execute(
                    "SELECT content FROM thoughts "
                    "ORDER BY timestamp DESC LIMIT ?",
                    (limit,)
                )
                return [row[0] for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Failed to get recent memories: {e}")
            return []

    def search_similar(self, query: str, limit: int = 3) -> List[str]:
        """
        Semantic search for similar thoughts
        
        Args:
            query: Search query text
            limit: Maximum results to return
            
        Returns:
            List of similar thoughts
        """
        try:
            # Get all embeddings from DB
            with self.lock:
                cursor = self.conn.cursor()
                cursor.execute(
                    "SELECT id, embedding FROM thoughts "
                    "WHERE embedding IS NOT NULL"
                )
                rows = cursor.fetchall()
            
            if not rows:
                return []
                
            # Prepare FAISS index
            ids = [row[0] for row in rows]
            embeddings = [np.frombuffer(row[1], dtype=np.float32) for row in rows]
            embeddings = np.array(embeddings)
            
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(embeddings)
            
            # Search
            query_embed = self.embedder.encode([query])
            distances, indices = index.search(query_embed, limit)
            
            # Get matching thoughts
            with self.lock:
                cursor = self.conn.cursor()
                cursor.execute(
                    f"SELECT content FROM thoughts "
                    f"WHERE id IN ({','.join(map(str, [ids[i] for i in indices[0]]))})"
                )
                return [row[0] for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []

    def consolidate_memories(self, salience_threshold: float = 0.7):
        """
        Process high-salience memories into long-term storage
        
        Args:
            salience_threshold: Minimum salience for consolidation
        """
        try:
            # Get high-salience thoughts
            with self.lock:
                cursor = self.conn.cursor()
                cursor.execute(
                    "SELECT content FROM thoughts "
                    "WHERE salience >= ? "
                    "ORDER BY timestamp DESC",
                    (salience_threshold,)
                )
                important = [row[0] for row in cursor.fetchall()]
            
            if not important:
                return
                
            # Here you could add:
            # - Clustering similar thoughts
            # - LLM summarization
            # - Knowledge graph updates
            logger.info(f"Found {len(important)} important memories to consolidate")
            
        except Exception as e:
            logger.error(f"Consolidation failed: {e}")

    def __del__(self):
        """Cleanup on destruction"""
        try:
            self.conn.close()
            logger.info("MemoryManager closed database connection")
        except:
            pass

# Test function
def test_memory_manager():
    """Run basic functionality tests"""
    import tempfile
    with tempfile.NamedTemporaryFile() as tmp:
        mm = MemoryManager(tmp.name)
        
        # Test adding thoughts
        assert mm.add_thought("Test thought 1", 0.8)
        assert mm.add_thought("Test thought 2", 0.9, "test")
        
        # Test retrieval
        assert len(mm.get_recent_memories(2)) == 2
        assert "Test thought 2" in mm.get_recent_memories(2)
        
        # Test search
        assert len(mm.search_similar("test", 1)) > 0
        
        print("All basic tests passed!")

if __name__ == "__main__":
    test_memory_manager()