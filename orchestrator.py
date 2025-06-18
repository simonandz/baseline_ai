import os
import time
import queue
import threading
import logging
import sys
from datetime import datetime

import torch
from transformers import pipeline
from conscious.pipeline import ConsciousProcessor
from subconscious.mind import Subconscious
from memory.manager import MemoryManager
from memory.preload import initialize_base_knowledge
from conversation.bus import ConversationBus, Message, Role

IDLE_THRESHOLD = 5.0  # seconds of user activity before day–dreaming may resume

# ──────────────────────────────────────────────────────────────────────────────
#  Enhanced CUDA / logging setup
# ──────────────────────────────────────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Reduce TensorFlow verbosity
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # Disable for better async

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("orchestrator")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.WARNING)

# ──────────────────────────────────────────────────────────────────────────────
#  Memory-optimized device setup
# ──────────────────────────────────────────────────────────────────────────────
# In orchestrator.py, modify the setup_device function:
def setup_device():
    try:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            # Set manual seed for reproducibility
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            return device
        else:
            device = torch.device("cpu")
            logger.info("Using CPU")
            return device
    except Exception as e:
        logger.error(f"Device initialization failed: {str(e)}")
        return torch.device("cpu")

device = setup_device()

# ──────────────────────────────────────────────────────────────────────────────
#  CPU-only chat model with memory limits
# ──────────────────────────────────────────────────────────────────────────────
def create_chat_model():
    """Create chat model with memory constraints"""
    try:
        logger.info("Loading DistilGPT2 chat model...")
        return pipeline(
            "text-generation",
            model="distilgpt2",
            device=-1,  # force CPU
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
        )
    except Exception as e:
        logger.error(f"Chat model load failed: {str(e)}")
        return None

_chat = create_chat_model()

def respond_to_user(user_text: str) -> str:
    """Safe response generation with fallbacks"""
    if not _chat:
        return "I'm initializing, please wait..."
    
    prompt = (
        "You are **Maddie**, an AI designed to converse naturally with a human.\n"
        "Do NOT echo or mirror the user's exact words.\n"
        f"User: {user_text}\n"
        "Maddie:"
    )
    try:
        out = _chat(
            prompt,
            max_new_tokens=60,
            temperature=0.6,
            top_p=0.9,
            pad_token_id=50256  # Specific to GPT-2 models
        )[0]["generated_text"]
        reply = out.replace(prompt, "").strip().split("\n")[0]
        return reply
    except Exception as e:
        logger.error(f"Response generation failed: {str(e)}")
        return "I'm having trouble processing that right now."

# ──────────────────────────────────────────────────────────────────────────────
#  MAIN with model loading fixes
# ──────────────────────────────────────────────────────────────────────────────
def main(bus: ConversationBus | None = None) -> None:
    try:
        if bus is None:
            bus = ConversationBus()

        # 1) Initialize memory first to reserve resources
        logger.info("Initializing memory system...")
        mem_manager = MemoryManager()
        initialize_base_knowledge(mem_manager)

        # 2) Print identity after memory init
        print("\n" + "="*40)
        print("AI Identity: Maddie")
        print("- I am Maddie, an artificial intelligence system designed to simulate human thought processes.")
        print("="*40 + "\n")

        # 3) Shared components with safe initialization
        thought_queue: queue.Queue[tuple[str,str]] = queue.Queue(maxsize=50)  # Smaller queue
        processor = ConsciousProcessor()

        # Timing variables
        last_consolidation = time.time()
        last_curiosity = time.time()
        last_user_time = time.time()

        # 4) Subconscious initialization with progress tracking
        logger.info("Initializing subconscious model...")
        subconscious = None
        try:
            subconscious = Subconscious(
                output_queue=thought_queue,
                memory=mem_manager,
                model_name="microsoft/phi-2",
                device=device,
            )
            
            # Start in separate thread with delayed loading
            load_thread = threading.Thread(
                target=subconscious.start,
                name="SubconsciousLoader"
            )
            load_thread.daemon = True
            load_thread.start()
            
            # Wait for model to partially load
            while not subconscious.is_ready() and load_thread.is_alive():
                logger.info("Loading model...")
                time.sleep(3)
                
            if not subconscious.is_ready():
                raise RuntimeError("Subconscious failed to initialize")
                
            logger.info("Subconscious model ready")
        except Exception as e:
            logger.error(f"Subconscious initialization failed: {str(e)}")
            subconscious = None

        # Main loop with model status check
        logger.info("Entering main operation loop")
        model_loaded = False
        
        while True:
            # Check if model finished loading
            if subconscious and not model_loaded and subconscious.is_ready():
                model_loaded = True
                logger.info("AI model fully loaded and operational")
                bus.outgoing.put(Message(
                    Role.SYSTEM, 
                    "System initialized and ready", 
                    time.time()
                ))
            
            # ─── 1) Process user messages ─────────────────────────────────────
            if not bus.incoming.empty():
                msg = bus.incoming.get()
                text = msg.text.strip()
                last_user_time = time.time()
                logger.info(f"User: {text}")

                mem_manager.add_memory(f"[USER] {text}", 0.6)
                thought_queue.put(("USER", text))

                # Immediate reply
                reply = respond_to_user(text)
                bus.outgoing.put(Message(Role.AGENT, reply, time.time()))
                mem_manager.add_memory(f"[MADDIE] {reply}", 0.6)
                logger.info(f"Reply: {reply}")

            # ─── 2) Process subconscious thoughts ─────────────────────────────
            if not thought_queue.empty():
                try:
                    item = thought_queue.get(timeout=0.5)
                    who, thought = item
                    logger.info(f"Processing {who} thought")
                    
                    if who == "AI":
                        result = processor.process(thought)
                        if result.get("passes", False):
                            refined = result.get("refined", thought)
                            bus.outgoing.put(Message(Role.AGENT, refined, time.time()))
                            mem_manager.add_memory(f"[MADDIE] {refined}", result.get("salience", 0.5))
                    thought_queue.task_done()
                except queue.Empty:
                    pass

            # ─── 3) Memory consolidation ──────────────────────────────────────
            if time.time() - last_consolidation > 3600:  # Hourly
                logger.info("Consolidating memory...")
                mem_manager.consolidate_memory()
                last_consolidation = time.time()

            # ─── 4) Curiosity injection ───────────────────────────────────────
            idle = time.time() - last_user_time
            if idle >= IDLE_THRESHOLD and time.time() - last_curiosity > 120:
                if subconscious and model_loaded:
                    question = "What aspect of my environment do I still not understand?"
                    thought_queue.put(("AI", question))
                    logger.info(f"Curiosity: {question}")
                    
                    # System snapshot
                    snap = f"System | Time: {datetime.utcnow().isoformat()}"
                    if torch.cuda.is_available():
                        props = torch.cuda.get_device_properties(0)
                        snap += f" | GPU: {props.name} | VRAM: {props.total_memory//(1024**3)}GB"
                    mem_manager.add_memory(snap, salient_score=0.3)
                    
                    last_curiosity = time.time()

            # ─── 5) Subconscious throttling ───────────────────────────────────
            if subconscious:
                if idle < IDLE_THRESHOLD:
                    subconscious.pause()
                else:
                    subconscious.resume()

            time.sleep(0.1)
            
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
    except Exception as e:
        logger.exception("Fatal error:")
    finally:
        logger.info("Cleaning up resources...")
        if subconscious:
            subconscious.stop()
        torch.cuda.empty_cache()
        logger.info("Shutdown complete")

if __name__ == "__main__":
    main()