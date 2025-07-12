class User:
    """Simple user class representing the current messaging user."""

    def __init__(self, name: str):
        self.name = name

    def send_message(self, to: str, message: str):
        """Send a message to the target user (prints to stdout)."""
        print(f"{self.name} sends '{message}' to {to}") 

class PremiumUser(User):
    def send_priority_message(self, to, message):
        print(f"{self.name} sends PRIORITY message '{message}' to {to}")