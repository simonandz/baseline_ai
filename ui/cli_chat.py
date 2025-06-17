import threading
import time
import importlib
from conversation import ConversationBus, Message, Role

# Create the shared bus
bus = ConversationBus()

def _agent_runner():
    # reload to pick up any edits to orchestrator.py
    import orchestrator
    importlib.reload(orchestrator)
    orchestrator.main(bus)

# Launch the agent in background
threading.Thread(target=_agent_runner, daemon=True).start()

print("🤖 Chat started. Type a message and press Enter (Ctrl-C to stop).")
try:
    while True:
        user_text = input("You: ")
        bus.incoming.put(Message(Role.USER, user_text, time.time()))
        # brief pause to let the agent think
        time.sleep(0.1)
        while not bus.outgoing.empty():
            msg = bus.outgoing.get()
            print("AI:", msg.text)
except KeyboardInterrupt:
    print("\nChat terminated.")
