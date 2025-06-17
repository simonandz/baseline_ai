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
from conversation import ConversationBus, Message, Role  # NEW

AGENT_READY = False          # global flag

# --------------------------------------------------------------------------- #
#  CUDA / logging boilerplate                                                 #
# --------------------------------------------------------------------------- #
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

# --------------------------------------------------------------------------- #
#  MAIN                                                                       #
# --------------------------------------------------------------------------- #
def main(bus: ConversationBus | None = None) -> None:
    # ----------------------------------------------------------------------- #
    #  Initialise shared resources                                            #
    # ----------------------------------------------------------------------- #
    if bus is None:
        bus = ConversationBus()

    mem_manager = MemoryManager()
    initialize_base_knowledge(mem_manager)

    print("\n" + "=" * 40)
    print("AI Identity: Maddie")
    print("- I am Maddie, an artificial intelligence system designed to simulate human thought processes.")
    print("=" * 40 + "\n")

    thought_queue: queue.Queue[str] = queue.Queue(maxsize=100)
    processor   = ConsciousProcessor()

    subconscious = Subconscious(
        output_queue=thought_queue,
        memory=mem_manager,
        model_name="microsoft/phi-2",
        device=device,
    )

    subconscious.start()
    print("Subconscious started")
    global AGENT_READY
    AGENT_READY = True


    # ----------------------------------------------------------------------- #
    #  Background console monitor                                             #
    # ----------------------------------------------------------------------- #
    def monitor_subconscious() -> None:
        print("\nSubconscious Monitor Active")
        while True:
            if not thought_queue.empty():
                t = thought_queue.get()
                ts = datetime.now().strftime("%H:%M:%S")
                print(f"🧠 [{ts}] {t}")
                thought_queue.task_done()
            time.sleep(0.1)

    threading.Thread(target=monitor_subconscious, daemon=True).start()

    # ----------------------------------------------------------------------- #
    #  Main event loop                                                        #
    # ----------------------------------------------------------------------- #
    last_consolidation = time.time()
    last_curiosity     = time.time()

    while True:
        # 1) ─── ingest USER messages ────────────────────────────────────────
        if not bus.incoming.empty():
            msg = bus.incoming.get()
            mem_manager.add_memory(f"[USER] {msg.text}", salient_score=0.6)
            thought_queue.put(msg.text)           # surface to subconscious

        # 2) ─── process a subconscious thought ─────────────────────────────
        if not thought_queue.empty():
            thought = thought_queue.get()
            result  = processor.process(thought)

            if result.get("passes", False):
                refined = result.get("refined", thought)
                print(f"💭 {refined}")

                mem_manager.add_memory(refined, salient_score=result.get("salience", 0.5))
                bus.outgoing.put(Message(Role.AGENT, refined, time.time()))

        # 3) ─── periodic memory consolidation ──────────────────────────────
        if time.time() - last_consolidation > 3600:  # every hour
            mem_manager.consolidate_memory()
            last_consolidation = time.time()
            print("Memory consolidation completed")

        # 4) ─── curiosity injection every 120 s ────────────────────────────
        if time.time() - last_curiosity > 120:
            thought_queue.put("What aspect of my environment do I still not understand?")

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


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
