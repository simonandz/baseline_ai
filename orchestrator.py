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

IDLE_THRESHOLD = 5.0  # seconds of silence before day-dreaming may resume

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
#  Small, CPU‐only chat model for conscious replies (to avoid OOM on GPU)
# ──────────────────────────────────────────────────────────────────────────────
_chat = pipeline(
    "text-generation",
    model="distilgpt2",
    device=-1,            # ← force CPU
)

def respond_to_user(user_text: str) -> str:
    prompt = (
        "You are **Maddie**, an AI designed to converse naturally with a human.\n"
        "Do NOT echo or mirror the user's exact words.\n"
        f"User: {user_text}\n"
        "Maddie:"
    )
    out = _chat(
        prompt,
        max_new_tokens=60,
        temperature=0.6,
        top_p=0.9
    )[0]["generated_text"]
    # strip off prompt
    reply = out.replace(prompt, "").strip().split("\n")[0]
    return reply

# ──────────────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main(bus: ConversationBus | None = None) -> None:
    if bus is None:
        bus = ConversationBus()

    # … unchanged initialisation …
    subconscious = Subconscious(
        output_queue=thought_queue,
        memory=mem_manager,
        model_name="microsoft/phi-2",
        device=device,
    )
    subconscious.start()

    # ─── new state variable ───────────────────────────────────────────
    last_user_time = time.time()

    # -----------------------------------------------------------------
    while True:
        # 1) ingest USER messages
        if not bus.incoming.empty():
            msg = bus.incoming.get()
            last_user_time = time.time()                 #  ← update
            mem_manager.add_memory(f"[USER] {msg.text}", 0.6)
            thought_queue.put(("USER", msg.text))

        # 2) conscious processing of any queued thought
        if not thought_queue.empty():
            who, thought = thought_queue.get()
            if who == "AI":
                result = processor.process(thought)
                if result.get("passes", False):
                    refined = result.get("refined", thought)
                    bus.outgoing.put(Message(Role.AGENT,
                                             refined,
                                             time.time()))
                    mem_manager.add_memory(refined,
                                           result.get("salience", 0.5))
            # ignore USER marker here
            thought_queue.task_done()

        # ────────────────────────
        # 3) hourly consolidation
        # ────────────────────────
        if time.time() - last_consolidation > 3600:
            mem.consolidate_memory()
            last_consolidation = time.time()

        # ────────────────────────
        # 4) inject curiosity
        # ────────────────────────
        if time.time() - last_curiosity > 120:
            question = "What aspect of my environment do I still not understand?"
            thought_queue.put(question)

            # snapshot GPU/clock for memory
            if torch.cuda.is_available():
                props     = torch.cuda.get_device_properties(0)
                snap      = f"EnvSnap | time={int(time.time())} | gpu={props.name} | vram={round(props.total_memory/(1024**3),1)}GB"
            else:
                snap      = f"EnvSnap | time={int(time.time())}"
            mem.add_memory(snap, salient_score=0.3)

            last_curiosity = time.time()
        # ─── 5) throttle subconscious ────────────────────────────────
        if time.time() - last_user_time < IDLE_THRESHOLD:
            subconscious.pause()     # user is active → suppress day-dreams
        else:
            subconscious.resume()    # user quiet → free to generate

        time.sleep(0.1)

if __name__ == "__main__":
    main()
