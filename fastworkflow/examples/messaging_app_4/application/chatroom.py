from .user import User, PremiumUser


class ChatRoom:
    def __init__(self):
        self.current_user = None
        self.users = []

    def add_user(self, user: User):
        self.users.append(user)

    def list_users(self) -> list[str]:
        return [user.name for user in self.users]

    @property
    def current_user(self) -> User:
        return self._current_user

    @current_user.setter
    def current_user(self, value: User):
        self._current_user = value

    def broadcast(self, message) -> None:
        sender_name = self._current_user.name if self._current_user else 'Anonymous'
        msg_priority = 'PRIORITY' if isinstance(self._current_user, PremiumUser) else ''

        if self.users:
            for user in self.users:
                if user.name == sender_name:
                    continue
                print(f"user {sender_name} is broadcasting {msg_priority} '{message}' to {user.name}")
        else:
            print("No users found in this chat room. Add some users first")
