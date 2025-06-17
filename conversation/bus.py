from queue import Queue
from dataclasses import dataclass
from enum import Enum, auto
import time

class Role(Enum):
    USER  = auto()
    AGENT = auto()

@dataclass
class Message:
    role: Role
    text: str
    ts:   float  # unix timestamp

class ConversationBus:
    """Thread-safe message hub for UI ↔ agent communication."""
    def __init__(self, maxsize: int = 500):
        self.incoming: Queue[Message] = Queue(maxsize=maxsize)   # UI → agent
        self.outgoing: Queue[Message] = Queue(maxsize=maxsize)   # agent → UI
