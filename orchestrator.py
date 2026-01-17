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

# Configuration
IDLE_THRESHOLD = 5.0  # seconds of user inactivity before daydreaming resumes
MODEL_LOAD_TIMEOUT = 120  # seconds to wait for model loading

# Environment setup
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logs
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Disable CUDA blocking

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("orchestrator")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.WARNING)

def setup_device():
    """Initialize device with memory optimizations"""
    try:
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.cuda.empty_cache()
            device = torch.device("cuda")
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"VRAM: {torch.cuda.memory_allocated()/1024**2:.1f}MB used / {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB total")
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

def create_chat_model():
    """Create chat model with memory constraints"""
    try:
        logger.info("Loading DistilGPT2 chat model...")
        return pipeline(
            "text-generation",
            model="distilgpt2",
            device=-1,  # force CPU
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
    except Exception as e:
        logger.error(f"Chat model load failed: {str(e)}")
        return None

def main(bus: ConversationBus | None = None) -> None:
    try:
        if bus is None:
            bus = ConversationBus()

        # Initialize components
        device = setup_device()
        _chat = create_chat_model()

        # 1) Initialize memory system
        logger.info("Initializing memory system...")
        mem_manager = MemoryManager()
        initialize_base_knowledge(mem_manager)

        # 2) Print identity
        print("\n" + "="*40)
        print("AI Identity: Maddie")
        print("- I am Maddie, an artificial intelligence system designed to simulate human thought processes.")
        print("="*40 + "\n")

        # 3) Shared components
        thought_queue = queue.Queue(maxsize=50)
        processor = ConsciousProcessor()

        # Timing variables
        last_consolidation = time.time()
        last_curiosity = time.time()
        last_user_time = time.time()

        # 4) Subconscious initialization
        logger.info("Initializing subconscious model...")
        subconscious = None
        try:
            subconscious = Subconscious(
                output_queue=thought_queue,
                memory=mem_manager,
                device=device
            )
            
            # Start in separate thread
            subconscious.start()
            
            # Wait for model to load
            start_time = time.time()
            while not subconscious.is_ready():
                if time.time() - start_time > MODEL_LOAD_TIMEOUT:
                    raise TimeoutError("Subconscious model loading timed out")
                logger.info("Waiting for model to load...")
                time.sleep(3)
                
            logger.info("Subconscious model ready")
        except Exception as e:
            logger.error(f"Subconscious initialization failed: {str(e)}")
            subconscious = None

        # Main operation loop
        logger.info("Entering main operation loop")
        subconscious_state = "paused"  # Start in paused state
        while True:
            current_time = time.time()
            
            # 1) Process user messages
            if not bus.incoming.empty():
                msg = bus.incoming.get()
                text = msg.text.strip()
                last_user_time = current_time
                logger.info(f"User: {text}")

                mem_manager.add_memory(f"[USER] {text}", 0.6)
                thought_queue.put(("USER", text))

                # Immediate reply
                reply = _chat(
                    f"You are Maddie. Respond to this naturally:\nUser: {text}\nMaddie:",
                    max_new_tokens=60,
                    temperature=0.6,
                    top_p=0.9,
                    pad_token_id=50256
                )[0]["generated_text"].split("Maddie:")[1].strip()
                
                bus.outgoing.put(Message(Role.AGENT, reply, time.time()))
                mem_manager.add_memory(f"[MADDIE] {reply}", 0.6)
                logger.info(f"Reply: {reply}")

            # 2) Process subconscious thoughts
            if not thought_queue.empty():
                try:
                    who, thought = thought_queue.get(timeout=0.5)
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

            # 3) Memory consolidation
            if current_time - last_consolidation > 3600:  # Hourly
                logger.info("Consolidating memory...")
                mem_manager.consolidate_memory()
                last_consolidation = current_time

            # 4) Curiosity injection
            idle = current_time - last_user_time
            if idle >= IDLE_THRESHOLD and current_time - last_curiosity > 120:
                if subconscious and subconscious.is_ready():
                    question = "What should I think about next?"
                    thought_queue.put(("AI", question))
                    logger.info(f"Curiosity: {question}")
                    
                    # System snapshot
                    snap = f"System | Time: {datetime.utcnow().isoformat()}"
                    if torch.cuda.is_available():
                        props = torch.cuda.get_device_properties(0)
                        snap += f" | GPU: {props.name} | VRAM: {props.total_memory//(1024**3)}GB"
                    mem_manager.add_memory(snap, salient_score=0.3)
                    
                    last_curiosity = current_time

            # 5) Subconscious throttling
            if subconscious:
                idle = current_time - last_user_time
                desired_state = "running" if idle >= IDLE_THRESHOLD else "paused"
                
                # Only change state if it's different
                if desired_state != subconscious_state:
                    if desired_state == "running":
                        logger.info("Resuming subconscious (user idle)")
                        subconscious.resume()
                    else:
                        logger.info("Pausing subconscious (user active)")
                        subconscious.pause()
                    subconscious_state = desired_state

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