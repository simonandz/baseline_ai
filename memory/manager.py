# memory/manager.py
import sqlite3
import numpy as np
import faiss
import threading
import queue
import logging
from datetime import datetime
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class MemoryManager:
    def __init__(self, db_path: str = "memory/mind.db"):
        self.db_path = db_path
        self.connection_pool = queue.Queue(maxsize=5)
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self._initialize_connections()
        self._init_db()
        logger.info("MemoryManager initialized with connection pool")

    def _initialize_connections(self):
        """Create initial connection pool"""
        for _ in range(3):
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.connection_pool.put(conn)

    def _get_connection(self):
        """Get a database connection from pool"""
        return self.connection_pool.get()

    def _return_connection(self, conn):
        """Return connection to pool"""
        self.connection_pool.put(conn)

    def _init_db(self):
        """Initialize database schema"""
        conn = self._get_connection()
        try:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY,
                timestamp TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding BLOB,
                type TEXT NOT NULL,
                salience REAL DEFAULT 0.5,
                immutable BOOLEAN DEFAULT 0,
                parent_id INTEGER REFERENCES memories(id),
                coherence_score REAL DEFAULT 0.0
            )""")
            
            conn.execute("""
            CREATE TABLE IF NOT EXISTS relationships (
                source_id INTEGER NOT NULL REFERENCES memories(id),
                target_id INTEGER NOT NULL REFERENCES memories(id),
                relationship_type TEXT NOT NULL,
                strength REAL DEFAULT 1.0,
                PRIMARY KEY (source_id, target_id, relationship_type)
            )""")
            
            conn.commit()
        finally:
            self._return_connection(conn)

    def add_memory(
        self,
        content: str,
        memory_type: str,
        salience: float = 0.5,
        parent_id: Optional[int] = None,
        immutable: bool = False
    ) -> int:
        """Add a coherent memory with relational context"""
        conn = self._get_connection()
        try:
            # Generate embedding
            embedding = self.embedder.encode([content])[0]
            
            # Calculate coherence with parent
            coherence = self._calculate_coherence(content, parent_id, conn) if parent_id else 0.7
            
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO memories 
                (timestamp, content, embedding, type, salience, immutable, parent_id, coherence_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now().isoformat(),
                    content,
                    embedding.tobytes(),
                    memory_type,
                    salience,
                    int(immutable),
                    parent_id,
                    coherence
                )
            )
            memory_id = cursor.lastrowid
            
            # Create relationships
            if parent_id:
                cursor.execute(
                    """INSERT INTO relationships 
                    (source_id, target_id, relationship_type, strength)
                    VALUES (?, ?, ?, ?)""",
                    (memory_id, parent_id, "hierarchical", min(1.0, coherence + 0.2))
                )
            
            conn.commit()
            return memory_id
        except Exception as e:
            logger.error(f"Add memory failed: {e}")
            raise
        finally:
            self._return_connection(conn)

    def _calculate_coherence(self, content: str, parent_id: int, conn: sqlite3.Connection) -> float:
        """Calculate how well this memory fits with parent memory"""
        cursor = conn.cursor()
        cursor.execute(
            "SELECT content, embedding FROM memories WHERE id = ?",
            (parent_id,)
        )
        parent_content, parent_embed = cursor.fetchone()
        parent_embed = np.frombuffer(parent_embed)
        
        current_embed = self.embedder.encode([content])[0]
        similarity = np.dot(parent_embed, current_embed) / (
            np.linalg.norm(parent_embed) * np.linalg.norm(current_embed) + 1e-8)
        
        # Content similarity
        content_sim = len(set(content.split()) & set(parent_content.split())) / max(
            len(set(content.split())), 1)
        
        return min(1.0, (similarity + content_sim) / 2)

    def get_recent_memories(self, limit: int = 5) -> List[Dict]:
        """Get most recent memories with context"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT id, content, type, salience 
                FROM memories 
                WHERE immutable = 0
                ORDER BY timestamp DESC 
                LIMIT ?""",
                (limit,)
            )
            
            memories = []
            for row in cursor.fetchall():
                mem = {
                    "id": row[0],
                    "content": row[1],
                    "type": row[2],
                    "salience": row[3]
                }
                
                # Get parent context
                cursor.execute(
                    """SELECT content 
                    FROM memories 
                    WHERE id = (
                        SELECT source_id 
                        FROM relationships 
                        WHERE target_id = ? AND relationship_type = 'hierarchical'
                    )""",
                    (mem["id"],)
                )
                parent = cursor.fetchone()
                if parent:
                    mem["context"] = parent[0]
                
                memories.append(mem)
                
            return memories
        finally:
            self._return_connection(conn)

    def get_core_identity(self) -> Dict:
        """Retrieve Maddie's core identity facts"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT content 
                FROM memories 
                WHERE type = 'identity' 
                ORDER BY salience DESC"""
            )
            identity = {"core": [], "capabilities": []}
            
            for row in cursor.fetchall():
                identity["core"].append(row[0])
                
            # Get related capabilities
            cursor.execute(
                """SELECT m.content 
                FROM memories m
                JOIN relationships r ON m.id = r.target_id
                JOIN memories parent ON r.source_id = parent.id
                WHERE parent.type = 'identity' 
                AND r.relationship_type = 'hierarchical'"""
            )
            for row in cursor.fetchall():
                identity["capabilities"].append(row[0])
                
            return identity
        finally:
            self._return_connection(conn)

    def find_related_memories(self, content: str, limit: int = 3) -> List[Dict]:
        """Semantic search for related memories"""
        conn = self._get_connection()
        try:
            # Get all embeddings
            cursor = conn.execute("SELECT id, embedding FROM memories")
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
            query_embed = self.embedder.encode([content])
            distances, indices = index.search(query_embed, limit)
            
            # Retrieve memories
            placeholders = ','.join(['?']*len(indices[0]))
            cursor = conn.execute(
                f"""SELECT id, content, type, salience 
                FROM memories 
                WHERE id IN ({placeholders}) 
                ORDER BY salience DESC""",
                [ids[i] for i in indices[0]]
            )
            
            return [{
                "id": row[0],
                "content": row[1],
                "type": row[2],
                "salience": row[3]
            } for row in cursor.fetchall()]
        finally:
            self._return_connection(conn)

    def consolidate_insights(self):
        """Generate higher-level insights from recent memories"""
        # Implementation would:
        # 1. Cluster related observations
        # 2. Generate abstract patterns
        # 3. Create new insight memories
        pass