import os
import time
import queue
import threading
import logging
import sys  # Added for exception handling
from datetime import datetime

import torch
from transformers import pipeline
from conscious.pipeline import ConsciousProcessor
from subconscious.mind import Subconscious
from memory.manager import MemoryManager
from memory.preload import initialize_base_knowledge
from conversation.bus import ConversationBus, Message, Role  # Fixed import path

IDLE_THRESHOLD = 5.0  # seconds of user activity before day–dreaming may resume

# ──────────────────────────────────────────────────────────────────────────────
#  CUDA / logging boilerplate
# ──────────────────────────────────────────────────────────────────────────────
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.basicConfig(level=logging.INFO)  # Added for better debugging
logger = logging.getLogger(__name__)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.WARNING)

# ──────────────────────────────────────────────────────────────────────────────
#  Device setup with better error handling
# ──────────────────────────────────────────────────────────────────────────────
device = None
try:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
except Exception as e:
    logger.error(f"Device initialization failed: {str(e)}")
    device = torch.device("cpu")

# ──────────────────────────────────────────────────────────────────────────────
#  Small, CPU-only chat model for immediate user replies
# ──────────────────────────────────────────────────────────────────────────────
try:
    _chat = pipeline(
        "text-generation",
        model="distilgpt2",
        device=device if device.type == "cpu" else -1,  # Ensure CPU usage
    )
    logger.info("DistilGPT2 chat model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load chat model: {str(e)}")
    # Fallback to simple responses
    def respond_to_user(user_text: str) -> str:
        return "I'm processing your message..."
else:
    def respond_to_user(user_text: str) -> str:
        prompt = (
            "You are **Maddie**, an AI designed to converse naturally with a human.\n"
            "Do NOT echo or mirror the user's exact words.\n"
            f"User: {user_text}\n"
            "Maddie:"
        )
        try:
            out = _chat(prompt, max_new_tokens=60, temperature=0.6, top_p=0.9)[0]["generated_text"]
            # strip off prompt
            reply = out.replace(prompt, "").strip().split("\n")[0]
            return reply
        except Exception as e:
            logger.error(f"Chat generation failed: {str(e)}")
            return "I encountered an error processing your message."

# ──────────────────────────────────────────────────────────────────────────────
#  MAIN with improved error handling and diagnostics
# ──────────────────────────────────────────────────────────────────────────────
def main(bus: ConversationBus | None = None) -> None:
    try:
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

        # 4) Initialize subconscious with better diagnostics
        logger.info("Initializing subconscious...")
        try:
            subconscious = Subconscious(
                output_queue=thought_queue,
                memory       = mem_manager,
                model_name   = "microsoft/phi-2",
                device       = device,
            )
            subconscious.start()
            logger.info("Subconscious thread started")
        except Exception as e:
            logger.error(f"Subconscious initialization failed: {str(e)}")
            # Attempt to run without subconscious
            subconscious = None

        # ──────────────────────────────────────────────────────────────────────────
        logger.info("Entering main loop")
        while True:
            # ───── 1) ingest USER messages ─────────────────────────────────────
            if not bus.incoming.empty():
                msg = bus.incoming.get()
                text = msg.text.strip()
                last_user_time = time.time()
                logger.info(f"Received user message: {text}")

                # store in memory
                mem_manager.add_memory(f"[USER] {text}", 0.6)

                # enqueue for subconscious to process
                thought_queue.put(("USER", text))

                # immediate reply
                reply = respond_to_user(text)
                bus.outgoing.put(Message(Role.AGENT, reply, time.time()))
                mem_manager.add_memory(f"[MADDIE] {reply}", 0.6)
                logger.info(f"Sent reply: {reply}")

            # ───── 2) process next thought from subconscious ───────────────────
            if not thought_queue.empty():
                item = thought_queue.get()
                # uniform unpack
                who, thought = item
                logger.info(f"Processing thought from {who}: {thought}")
                if who == "AI":
                    try:
                        result = processor.process(thought)
                        if result.get("passes", False):
                            refined = result.get("refined", thought)
                            bus.outgoing.put(Message(Role.AGENT, refined, time.time()))
                            mem_manager.add_memory(f"[MADDIE] {refined}", result.get("salience", 0.5))
                            logger.info(f"Refined output: {refined}")
                    except Exception as e:
                        logger.error(f"Conscious processing failed: {str(e)}")
                # USER‐origin items are only enqueued for context, never re‐spoken
                thought_queue.task_done()

            # ───── 3) hourly memory consolidation ──────────────────────────────
            if time.time() - last_consolidation > 3600:
                logger.info("Performing memory consolidation")
                mem_manager.consolidate_memory()
                last_consolidation = time.time()

            # ───── 4) curiosity injection when idle ────────────────────────────
            idle = time.time() - last_user_time
            if idle >= IDLE_THRESHOLD and time.time() - last_curiosity > 120:
                question = "What aspect of my environment do I still not understand?"
                thought_queue.put(("AI", question))
                logger.info(f"Injecting curiosity: {question}")

                # take a little snapshot for memory
                if torch.cuda.is_available():
                    props = torch.cuda.get_device_properties(0)
                    snap = f"EnvSnap | time={int(time.time())} | gpu={props.name} | vram={round(props.total_memory/(1024**3),1)}GB"
                else:
                    snap = f"EnvSnap | time={int(time.time())}"
                mem_manager.add_memory(snap, salient_score=0.3)

                last_curiosity = time.time()

            # ───── 5) throttle subconscious activity ───────────────────────────
            if subconscious:
                if idle < IDLE_THRESHOLD:
                    subconscious.pause()
                else:
                    subconscious.resume()

            time.sleep(0.1)
            
    except Exception as e:
        logger.exception("Fatal error in main loop:")
        if subconscious:
            subconscious.stop()
        sys.exit(1)

if __name__ == "__main__":
    main()