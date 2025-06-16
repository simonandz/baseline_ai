import threading
import time
import numpy as np
from transformers import pipeline
from sentence_transformers import SentenceTransformer

class Subconscious:
    def __init__(self, output_queue, memory=None):
        self.queue = output_queue
        self.memory = memory
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.generator = pipeline(
            "text-generation",
            model="EleutherAI/gpt-neo-125M",
            device=0 if torch.cuda.is_available() else -1
        )
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run)

    def _generate_thought(self) -> str:
        """Generate with context awareness"""
        context = self._get_context()
        prompt = f"{context}\nNew thought:"
        
        for _ in range(3):  # Retry if duplicate
            result = self.generator(
                prompt,
                max_length=50,
                temperature=0.8,
                do_sample=True
            )
            thought = result[0]['generated_text'].strip()
            
            if not self._is_duplicate(thought):
                return thought
        return ""

    def _get_context(self) -> str:
        """Get relevant context from memory"""
        if not self.memory:
            return ""
        return "\n".join(self.memory.get_recent(2))

    def _is_duplicate(self, thought: str) -> bool:
        """Embedding-based duplicate check"""
        embedding = self.embedder.encode([thought])[0]
        if not hasattr(self, 'recent_embeddings'):
            self.recent_embeddings = []
            
        if self.recent_embeddings:
            similarities = np.dot(self.recent_embeddings, embedding)
            if np.max(similarities) > 0.85:
                return True
                
        self.recent_embeddings.append(embedding)
        if len(self.recent_embeddings) > 100:
            self.recent_embeddings.pop(0)
        return False

    def _run(self):
        while not self._stop_event.is_set():
            thought = self._generate_thought()
            if thought:
                self.queue.put(thought)
                if self.memory:
                    self.memory.add_thought(thought, salience=0.5)
            time.sleep(3)  # Interval

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join()