import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict, Tuple
from .config import *

class ThoughtFilter:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.recent_thoughts = []
        self.category_vectors = self._init_category_vectors()
    
    def _init_category_vectors(self) -> Dict[str, np.ndarray]:
        """Create TF-IDF vectors for each thought category"""
        category_docs = {}
        for category, keywords in THOUGHT_CATEGORIES.items():
            # Create pseudo-documents from keywords
            category_docs[category] = " ".join(keywords)
        
        # Fit TF-IDF to categories
        corpus = list(category_docs.values())
        tfidf_matrix = self.vectorizer.fit_transform(corpus)
        return {
            category: tfidf_matrix[i].toarray()[0]
            for i, category in enumerate(THOUGHT_CATEGORIES)
        }
    
    def _calculate_salience(self, thought: str) -> float:
        """Calculate thought importance score (0-1)"""
        score = 0.0
        
        # Length bonus (longer thoughts are more salient)
        word_count = len(thought.split())
        score += min(0.2, word_count / 50)
        
        # Keyword bonus
        lower_thought = thought.lower()
        for keyword in SALIENT_KEYWORDS:
            if keyword in lower_thought:
                score += 0.1
                break  # Max one keyword bonus
        
        # Question bonus
        if "?" in thought:
            score += 0.15
        
        # Personal pronoun bonus (self-referential)
        if re.search(r"\b(I|me|my|mine)\b", thought, re.IGNORECASE):
            score += 0.1
        
        return min(1.0, score)
    
    def _calculate_novelty(self, thought: str) -> float:
        """Calculate how different this thought is from recent ones"""
        if not self.recent_thoughts:
            return 1.0  # First thought is maximally novel
        
        # Update with recent thoughts
        docs = self.recent_thoughts + [thought]
        tfidf_matrix = self.vectorizer.fit_transform(docs)
        
        # Calculate similarity to most recent thought
        similarity = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[-2])[0][0]
        return 1.0 - similarity
    
    def _calculate_relevance(self, thought: str) -> Tuple[float, str]:
        """Determine relevance to current context and categorize"""
        thought_vec = self.vectorizer.transform([thought]).toarray()[0]
        
        best_category = "misc"
        best_score = 0.0
        
        # Compare to each category vector
        for category, cat_vec in self.category_vectors.items():
            similarity = cosine_similarity([thought_vec], [cat_vec])[0][0]
            if similarity > best_score:
                best_score = similarity
                best_category = category
        
        return best_score, best_category
    
    def evaluate_thought(self, thought: str) -> Dict:
        """Comprehensive thought evaluation"""
        # Update recent thoughts (for novelty calculation)
        self.recent_thoughts.append(thought)
        if len(self.recent_thoughts) > 20:
            self.recent_thoughts.pop(0)
        
        salience = self._calculate_salience(thought)
        novelty = self._calculate_novelty(thought)
        relevance, category = self._calculate_relevance(thought)
        
        return {
            "raw": thought,
            "salience": salience,
            "novelty": novelty,
            "relevance": relevance,
            "category": category,
            "passes": (
                salience >= SALIENCE_THRESHOLD and
                novelty >= NOVELTY_THRESHOLD and
                relevance >= RELEVANCE_THRESHOLD
            )
        }