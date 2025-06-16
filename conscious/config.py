# Configuration for conscious thought processing
CONSCIOUS_CHECK_INTERVAL = 1.0  # seconds
SALIENCE_THRESHOLD = 0.65  # Minimum importance score
NOVELTY_THRESHOLD = 0.55   # Minimum novelty score
RELEVANCE_THRESHOLD = 0.6   # Minimum context relevance

# Refinement model (smaller/faster than subconscious model)
REFINEMENT_MODEL = "facebook/bart-large-cnn"
REFINEMENT_MAX_TOKENS = 80
REFINEMENT_TEMPERATURE = 0.5  # More deterministic for refinement

# Salience keywords (thoughts containing these get bonus)
SALIENT_KEYWORDS = [
    "why", "how", "important", "remember", "idea", "solution",
    "problem", "question", "insight", "realize", "understand",
    "significance", "implication", "connection"
]

# Thought categories for semantic organization
THOUGHT_CATEGORIES = {
    "problem_solving": ["solution", "fix", "solve", "debug", "error"],
    "planning": ["plan", "schedule", "next", "later", "tomorrow"],
    "insight": ["realize", "understand", "meaning", "significance"],
    "curiosity": ["why", "how", "wonder", "question"],
    "memory": ["remember", "recall", "forgot", "nostalgia"],
    "emotion": ["feel", "happy", "sad", "excited", "worried"]
}