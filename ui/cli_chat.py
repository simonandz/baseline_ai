# ui/cli_chat.py

import threading
import time
import importlib

from conversation import ConversationBus, Message, Role
import orchestrator  # ensures AGENT_READY flag is in scope

# Create the shared bus
bus = ConversationBus()

def _agent_runner():
    # reload to pick up any edits to orchestrator.py
    import orchestrator as orch
    importlib.reload(orch)
    orch.main(bus)

# Launch the agent in background
threading.Thread(target=_agent_runner, daemon=True).start()

# Wait until the agent signals it’s ready
print("Loading AI model … this may take a minute ⏳")
while not getattr(orchestrator, "AGENT_READY", False):
    time.sleep(0.5)

print("🤖 Chat ready! Type a message and press Enter (Ctrl-C to stop).")
try:
    while True:
        user_text = input("You: ")
        bus.incoming.put(Message(Role.USER, user_text, time.time()))

        # Give Maddie a moment to generate
        time.sleep(0.1)

        # Print out all available AI messages
        while not bus.outgoing.empty():
            msg = bus.outgoing.get()
            print("AI:", msg.text)

except KeyboardInterrupt:
    print("\nChat terminated.")
