# colab_interactive.py
"""
Interactive Colab runner for Maddie AI.
Integrates subconscious thought generation, conscious filtering, and user chat.

Usage in Colab:
    from colab_interactive import MaddieSession
    session = MaddieSession()
    session.run()
"""
import os
import time
import queue
import threading
import logging
import sys
from datetime import datetime
from typing import Optional
from IPython.display import display, HTML, clear_output
import ipywidgets as widgets

import torch
from transformers import pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("maddie")
logging.getLogger("transformers").setLevel(logging.ERROR)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class MaddieSession:
    """
    Interactive session manager for Maddie AI in Google Colab.

    Features:
    - Background subconscious thought generation (Phi-3)
    - Conscious filtering and refinement
    - Real-time user chat
    - Memory persistence
    - Thought display with streaming updates
    """

    def __init__(self, enable_conscious: bool = True):
        """
        Initialize Maddie session.

        Args:
            enable_conscious: Whether to enable conscious filtering/refinement
        """
        self.enable_conscious = enable_conscious
        self._stop_event = threading.Event()
        self._thought_history = []
        self._chat_history = []

        # Initialize device
        self.device = self._setup_device()

        # Initialize components
        logger.info("Initializing memory system...")
        from memory.manager import MemoryManager
        from memory.preload import initialize_base_knowledge
        self.memory = MemoryManager()
        initialize_base_knowledge(self.memory)

        # Chat model (lightweight for responses)
        logger.info("Loading chat model...")
        self._chat_model = pipeline(
            "text-generation",
            model="distilgpt2",
            device=-1,  # CPU for chat to save GPU for subconscious
            torch_dtype=torch.float32
        )

        # Conscious processor (optional)
        self.processor = None
        if enable_conscious:
            logger.info("Loading conscious processor...")
            from conscious.pipeline import ConsciousProcessor
            self.processor = ConsciousProcessor()

        # Subconscious
        logger.info("Initializing subconscious...")
        from subconscious.mind import Subconscious
        self.thought_queue = queue.Queue(maxsize=50)
        self.subconscious = Subconscious(
            output_queue=self.thought_queue,
            memory=self.memory,
            device=self.device
        )

        # UI components
        self._output_area = None
        self._input_box = None

        logger.info("Maddie session initialized")

    def _setup_device(self):
        """Initialize GPU if available."""
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.cuda.empty_cache()
            device = torch.device("cuda")
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            return device
        logger.info("Using CPU")
        return torch.device("cpu")

    def _format_thought(self, thought: str, thought_type: str = "subconscious") -> str:
        """Format a thought for display."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        if thought_type == "subconscious":
            return f'<div style="color: #6b7280; font-style: italic; margin: 4px 0;">[{timestamp}] 💭 {thought}</div>'
        elif thought_type == "conscious":
            return f'<div style="color: #3b82f6; margin: 4px 0;">[{timestamp}] 🧠 {thought}</div>'
        elif thought_type == "user":
            return f'<div style="color: #10b981; font-weight: bold; margin: 8px 0;">[{timestamp}] 👤 You: {thought}</div>'
        elif thought_type == "maddie":
            return f'<div style="color: #8b5cf6; margin: 8px 0;">[{timestamp}] 🤖 Maddie: {thought}</div>'
        return thought

    def _generate_response(self, user_input: str) -> str:
        """Generate a response to user input."""
        try:
            # Build context from recent memories
            recent = self.memory.get_recent_memories(3)
            context = "\n".join(recent) if recent else ""

            prompt = f"""You are Maddie, a thoughtful AI. Respond naturally and concisely.
Context: {context}
User: {user_input}
Maddie:"""

            result = self._chat_model(
                prompt,
                max_new_tokens=60,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=50256,
                do_sample=True
            )[0]["generated_text"]

            # Extract response after "Maddie:"
            if "Maddie:" in result:
                response = result.split("Maddie:")[-1].strip()
            else:
                response = result.replace(prompt, "").strip()

            # Clean up response
            response = response.split("\n")[0].strip()
            return response if response else "I'm thinking about that..."

        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return "I'm having trouble formulating a response right now."

    def _process_thought(self, thought: str) -> Optional[str]:
        """Process a subconscious thought through conscious filtering."""
        if not self.processor:
            return thought

        try:
            result = self.processor.process(thought)
            if result.get("passes", False):
                return result.get("refined", thought)
            return None
        except Exception as e:
            logger.error(f"Conscious processing failed: {e}")
            return thought

    def _update_display(self):
        """Update the output display with current thoughts."""
        if self._output_area is None:
            return

        # Build HTML from recent thoughts
        all_items = self._thought_history[-30:]  # Last 30 items

        html_content = "<div style='font-family: monospace; padding: 10px; background: #1a1a2e; color: #eee; border-radius: 8px;'>"
        if not all_items:
            html_content += "<p style='color: #888;'>Waiting for thoughts...</p>"
        else:
            html_content += "".join(all_items)
        html_content += "</div>"

        # Update widget (this should trigger Colab refresh)
        self._output_area.value = html_content

    def _on_user_input(self, widget):
        """Handle user input from the text box."""
        user_text = widget.value.strip()
        if not user_text:
            return

        # Clear input
        widget.value = ""

        # Pause subconscious during interaction
        self.subconscious.pause()

        # Add user message to display
        self._thought_history.append(self._format_thought(user_text, "user"))

        # Store in memory
        self.memory.add_memory(f"[USER] {user_text}", 0.6)

        # Generate response
        response = self._generate_response(user_text)

        # Add response to display
        self._thought_history.append(self._format_thought(response, "maddie"))

        # Store response in memory
        self.memory.add_memory(f"[MADDIE] {response}", 0.6)

        self._update_display()

        # Resume subconscious after delay
        threading.Timer(3.0, self.subconscious.resume).start()

    def _thought_processor_loop(self):
        """Background loop to process subconscious thoughts."""
        logger.info("Thought processor loop started")
        while not self._stop_event.is_set():
            try:
                # Check for new thoughts
                try:
                    who, thought = self.thought_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                logger.info(f"Processing thought from {who}: {thought[:50]}...")

                if who == "AI":
                    # Always show the raw thought first
                    self._thought_history.append(
                        self._format_thought(thought, "subconscious")
                    )

                    # Try conscious processing if enabled
                    if self.enable_conscious and self.processor:
                        try:
                            refined = self._process_thought(thought)
                            if refined and refined != thought:
                                self._thought_history.append(
                                    self._format_thought(f"[Refined] {refined}", "conscious")
                                )
                        except Exception as e:
                            logger.warning(f"Conscious processing failed: {e}")

                    # Store in memory
                    self.memory.add_memory(f"[THOUGHT] {thought}", 0.5)

                    self._update_display()

                self.thought_queue.task_done()

            except Exception as e:
                logger.error(f"Thought processing error: {e}")
                import traceback
                traceback.print_exc()

    def run(self):
        """
        Start the interactive session.
        Creates UI widgets and starts background processing.
        """
        print("\n" + "="*50)
        print("🧠 MADDIE AI - Interactive Session")
        print("="*50)
        print("Starting up... this may take a moment.\n")

        # Create UI
        self._output_area = widgets.HTML(
            value="<div style='font-family: monospace;'>Initializing...</div>",
            layout=widgets.Layout(width='100%', height='400px', overflow='auto')
        )

        self._input_box = widgets.Text(
            placeholder='Type a message to Maddie...',
            layout=widgets.Layout(width='100%')
        )
        self._input_box.on_submit(self._on_user_input)

        # Display UI
        display(widgets.VBox([
            widgets.HTML("<h3>💭 Maddie's Thoughts</h3>"),
            self._output_area,
            widgets.HTML("<h4>💬 Chat with Maddie</h4>"),
            self._input_box
        ]))

        # Start subconscious
        self.subconscious.start()

        # Wait for model to load
        print("Loading Phi-3 model...")
        while not self.subconscious.is_ready():
            time.sleep(1)
        print("Model ready! Thoughts will appear below.\n")

        # Start thought processor
        self._processor_thread = threading.Thread(
            target=self._thought_processor_loop,
            daemon=True
        )
        self._processor_thread.start()
        logger.info("Thought processor thread started")

        # Make sure subconscious is running (not paused)
        self.subconscious.resume()
        logger.info("Subconscious resumed")

        self._thought_history.append(
            self._format_thought("Session started. I'm beginning to think...", "conscious")
        )
        self._update_display()

    def stop(self):
        """Stop the session and clean up resources."""
        logger.info("Stopping session...")
        self._stop_event.set()
        self.subconscious.stop()
        torch.cuda.empty_cache()
        logger.info("Session stopped")

    def get_thoughts(self) -> list:
        """Get list of all generated thoughts."""
        return self._thought_history.copy()

    def get_memories(self, count: int = 10) -> list:
        """Get recent memories."""
        return self.memory.get_recent_memories(count)


def run_simple():
    """
    Simple non-interactive runner for testing.
    Prints thoughts to console without UI widgets.
    """
    print("\n" + "="*50)
    print("🧠 MADDIE AI - Console Mode")
    print("="*50 + "\n")

    # Initialize components
    from memory.manager import MemoryManager
    from memory.preload import initialize_base_knowledge
    from subconscious.mind import Subconscious

    memory = MemoryManager()
    initialize_base_knowledge(memory)

    thought_queue = queue.Queue(maxsize=50)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    subconscious = Subconscious(
        output_queue=thought_queue,
        memory=memory,
        device=device
    )

    subconscious.start()

    print("Waiting for model to load...")
    while not subconscious.is_ready():
        time.sleep(1)
    print("Model ready! Generating thoughts...\n")

    try:
        while True:
            try:
                who, thought = thought_queue.get(timeout=1)
                if who == "AI":
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    print(f"[{timestamp}] 💭 {thought}")
                thought_queue.task_done()
            except queue.Empty:
                continue
    except KeyboardInterrupt:
        print("\nStopping...")
        subconscious.stop()
        print("Done.")


if __name__ == "__main__":
    run_simple()
