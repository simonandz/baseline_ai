import os
import time
import queue
import threading
import logging
from datetime import datetime

import torch
from conscious.pipeline import ConsciousProcessor
from subconscious import Subconscious
from memory.manager import MemoryManager
from memory.preload import initialize_base_knowledge
from conversation import ConversationBus, Message, Role
from transformers import pipeline

_chat = pipeline(
    "text-generation",
    model="microsoft/phi-2",
    max_new_tokens=60,
    temperature=0.6,
    top_p=0.9,
)

def respond_to_user(user_text: str) -> str:
    prompt = (
        "You are Maddie, an AI conversing with a human.\n"
        f"Human: {user_text}\n"
        "Maddie:"
    )
    out = _chat(prompt)[0]["generated_text"].replace(prompt, "").strip()
    return out.split("\n")[0]  # first line


AGENT_READY = False  # flips True once models finish loading

# ---------------------------------------------------------------------------- #
#  CUDA / logging boilerplate                                                   
# ---------------------------------------------------------------------------- #
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

# ---------------------------------------------------------------------------- #
#  MAIN                                                                        
# ---------------------------------------------------------------------------- #
def main(bus: ConversationBus | None = None) -> None:
    global AGENT_READY

    # 1) Setup bus
    if bus is None:
        bus = ConversationBus()

    # 2) Initialize memory
    mem_manager = MemoryManager()
    initialize_base_knowledge(mem_manager)

    # 3) Identity banner
    print("\n" + "=" * 40)
    print("AI Identity: Maddie")
    print("- I am Maddie, an artificial intelligence system designed to simulate human thought processes.")
    print("=" * 40 + "\n")

    # 4) Prepare queue & processors
    thought_queue: queue.Queue[str | tuple[str, str]] = queue.Queue(maxsize=100)
    processor = ConsciousProcessor()
    subconscious = Subconscious(
        output_queue=thought_queue,
        memory=mem_manager,
        model_name="microsoft/phi-2",
        device=device,
    )

    # 5) Start subconscious
    subconscious.start()
    print("Subconscious started")
    AGENT_READY = True

    # ---------------------------------------------------------------------------- #
    #  Monitor thread – prints only AI-generated thoughts                           
    # ---------------------------------------------------------------------------- #
    def monitor_subconscious():
        print("\nSubconscious Monitor Active")
        while True:
            if not thought_queue.empty():
                item = thought_queue.get()
                thought_queue.task_done()
                # interpret plain string as AI
                who, txt = (item if isinstance(item, tuple) else ("AI", item))
                if who == "AI":
                    ts = datetime.now().strftime("%H:%M:%S")
                    print(f"🧠 [{ts}] {txt}")
            time.sleep(0.1)

    threading.Thread(target=monitor_subconscious, daemon=True).start()

    # ---------------------------------------------------------------------------- #
    #  Main loop – ingest user, process AI thoughts                                
    # ---------------------------------------------------------------------------- #
    last_consolidation = time.time()
    last_curiosity     = time.time()

    while True:
        # A) ingest USER messages
        if not bus.incoming.empty():
            msg = bus.incoming.get()

            # 1. store in memory (but DO NOT send to thought_queue)
            mem_manager.add_memory(f"[USER] {msg.text}", 0.8)

            # 2. immediate conversational reply
            reply = respond_to_user(msg.text)          # ← new helper (see below)
            bus.outgoing.put(Message(Role.AGENT, reply, time.time()))
            print("💬", reply)                         # optional console echo

        # B) process AI-generated subconscious thoughts
        if not thought_queue.empty():
            item = thought_queue.get(); thought_queue.task_done()
            if isinstance(item, str):  # subconscious always sends plain str
                result = processor.process(item)
                if result.get("passes"):
                    refined = result.get("refined", item)
                    mem_manager.add_memory(refined, result.get("salience", 0.5))
                    bus.outgoing.put(Message(Role.AGENT, refined, time.time()))


        # C) Hourly memory consolidation
        if time.time() - last_consolidation > 3600:
            mem_manager.consolidate_memory()
            last_consolidation = time.time()
            print("Memory consolidation completed")

        # D) Curiosity injection every 120s
        if time.time() - last_curiosity > 120:
            thought_queue.put(("AI", "What aspect of my environment do I still not understand?"))
            if torch.cuda.is_available():
                props     = torch.cuda.get_device_properties(0)
                gpu_name  = props.name
                gpu_memGB = round(props.total_memory / (1024**3), 1)
                env_fact  = f"EnvSnap | unix={int(time.time())} | gpu={gpu_name} | vram={gpu_memGB}GB"
            else:
                env_fact  = f"EnvSnap | unix={int(time.time())}"
            mem_manager.add_memory(env_fact, salient_score=0.3)
            last_curiosity = time.time()

        time.sleep(0.1)


# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
