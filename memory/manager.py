import sqlite3
import numpy as np
import faiss
from datetime import datetime
from sentence_transformers import SentenceTransformer

class MemoryManager:
    def __init__(self, db_path="memory/mind.db"):
        self.conn = sqlite3.connect(db_path)
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self._init_db()
        
    def _init_db(self):
        """Initialize database tables"""
        self.conn.execute("""
        CREATE TABLE IF NOT EXISTS thoughts (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            content TEXT,
            embedding BLOB,
            salience REAL,
            category TEXT
        )""")
        self.conn.commit()

    def add_thought(self, thought: str, salience: float):
        """Add thought with semantic embedding"""
        embedding = self.embedder.encode([thought])[0]
        self.conn.execute(
            "INSERT INTO thoughts VALUES (?, ?, ?, ?, ?, ?)",
            (None, datetime.now().isoformat(), thought, 
             embedding.tobytes(), salience, None)
        )
        self.conn.commit()

    def get_relevant(self, query: str, n=3) -> list:
        """Semantic search"""
        # Get all embeddings
        cur = self.conn.execute("SELECT embedding FROM thoughts")
        embeddings = [np.frombuffer(row[0], dtype=np.float32) for row in cur]
        if not embeddings:
            return []
            
        # Build index
        index = faiss.IndexFlatL2(embeddings[0].shape[0])
        index.add(np.array(embeddings))
        
        # Search
        query_embed = self.embedder.encode([query])
        distances, indices = index.search(query_embed, n)
        
        # Get results
        cur = self.conn.execute(
            "SELECT content FROM thoughts WHERE rowid IN ({})".format(
                ",".join([str(i+1) for i in indices[0]])
            )
        )
        return [row[0] for row in cur]