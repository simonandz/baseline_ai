import os
import time
import queue
import threading
import logging
from datetime import datetime

import torch
from transformers import pipeline
from conscious.pipeline import ConsciousProcessor
from subconscious.mind import Subconscious
from memory.manager import MemoryManager
from memory.preload import initialize_base_knowledge
from conversation import ConversationBus, Message, Role

IDLE_THRESHOLD = 5.0  # seconds of user activity before day–dreaming may resume

# ──────────────────────────────────────────────────────────────────────────────
#  CUDA / logging boilerplate
# ──────────────────────────────────────────────────────────────────────────────
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.WARNING)

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Using CPU")

# ──────────────────────────────────────────────────────────────────────────────
#  Small, CPU-only chat model for immediate user replies
# ──────────────────────────────────────────────────────────────────────────────
_chat = pipeline(
    "text-generation",
    model="distilgpt2",
    device=-1,            # force CPU
)

def respond_to_user(user_text: str) -> str:
    prompt = (
        "You are **Maddie**, an AI designed to converse naturally with a human.\n"
        "Do NOT echo or mirror the user's exact words.\n"
        f"User: {user_text}\n"
        "Maddie:"
    )
    out = _chat(prompt, max_new_tokens=60, temperature=0.6, top_p=0.9)[0]["generated_text"]
    # strip off prompt
    reply = out.replace(prompt, "").strip().split("\n")[0]
    return reply

# ──────────────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main(bus: ConversationBus | None = None) -> None:
    if bus is None:
        bus = ConversationBus()

    # 1) initialize memory
    mem_manager = MemoryManager()
    initialize_base_knowledge(mem_manager)

    # 2) print identity banner
    print("\n" + "="*40)
    print("AI Identity: Maddie")
    print("- I am Maddie, an artificial intelligence system designed to simulate human thought processes.")
    print("="*40 + "\n")

    # 3) shared thought‐queue and components
    thought_queue: queue.Queue[tuple[str,str]] = queue.Queue(maxsize=200)
    processor = ConsciousProcessor()

    last_consolidation = time.time()
    last_curiosity     = time.time()
    last_user_time     = time.time()

    subconscious = Subconscious(
        output_queue=thought_queue,
        memory       = mem_manager,
        model_name   = "microsoft/phi-2",
        device       = device,
    )
    subconscious.start()

    # ──────────────────────────────────────────────────────────────────────────
    while True:
        # ───── 1) ingest USER messages ─────────────────────────────────────
        if not bus.incoming.empty():
            msg = bus.incoming.get()
            text = msg.text.strip()
            last_user_time = time.time()

            # store in memory
            mem_manager.add_memory(f"[USER] {text}", 0.6)

            # enqueue for subconscious to chew on
            thought_queue.put(("USER", text))

            # immediate reply
            reply = respond_to_user(text)
            bus.outgoing.put(Message(Role.AGENT, reply, time.time()))
            mem_manager.add_memory(f"[MADDIE] {reply}", 0.6)

        # ───── 2) process next thought from subconscious ───────────────────
        if not thought_queue.empty():
            item = thought_queue.get()
            # uniform unpack
            who, thought = item
            if who == "AI":
                result = processor.process(thought)
                if result.get("passes", False):
                    refined = result.get("refined", thought)
                    bus.outgoing.put(Message(Role.AGENT, refined, time.time()))
                    mem_manager.add_memory(f"[MADDIE] {refined}", result.get("salience", 0.5))
            # USER‐origin items are only enqueued for context, never re‐spoken
            thought_queue.task_done()

        # ───── 3) hourly memory consolidation ──────────────────────────────
        if time.time() - last_consolidation > 3600:
            mem_manager.consolidate_memory()
            last_consolidation = time.time()

        # ───── 4) curiosity injection when idle ────────────────────────────
        idle = time.time() - last_user_time
        if idle >= IDLE_THRESHOLD and time.time() - last_curiosity > 120:
            question = "What aspect of my environment do I still not understand?"
            thought_queue.put(("AI", question))

            # take a little snapshot for memory
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                snap = f"EnvSnap | time={int(time.time())} | gpu={props.name} | vram={round(props.total_memory/(1024**3),1)}GB"
            else:
                snap = f"EnvSnap | time={int(time.time())}"
            mem_manager.add_memory(snap, salient_score=0.3)

            last_curiosity = time.time()

        # ───── 5) throttle subconscious activity ───────────────────────────
        if idle < IDLE_THRESHOLD:
            subconscious.pause()
        else:
            subconscious.resume()

        time.sleep(0.1)

if __name__ == "__main__":
    main()
