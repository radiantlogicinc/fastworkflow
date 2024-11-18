from queue import Queue, Empty
from threading import Lock

from fastworkflow.utils.logging import logger

class UserMessageQueues:
    def __init__(self):
        self._queues: dict[int, Queue] = {}
        self._lock = Lock()
   
    def get_queue(self, session_id: int) -> Queue:
        with self._lock:
            return self._queues[session_id]
 
    def add_queue(self, session_id: int) -> None:
        with self._lock:
            self._queues[session_id] = Queue()
    
    def remove_queue(self, session_id: int) -> None:
        """Remove a session's queue after draining"""
        with self._lock:
            if session_id not in self._queues:
                return
            
            # Drain any remaining messages
            remaining = self.drain_queue(session_id)
            if remaining:
                logger.warning(f"Removed queue for session {session_id} with {len(remaining)} pending messages")
            
            del self._queues[session_id]     
        
    def drain_queue(self, session_id: int) -> list[str]:
        """Drain all messages from a queue before removal"""
        with self._lock:
            if session_id not in self._queues:
                return []
            
            messages = []
            queue = self._queues[session_id]
            while not queue.empty():
                try:
                    messages.append(queue.get_nowait())
                except Empty:
                    break
            return messages
