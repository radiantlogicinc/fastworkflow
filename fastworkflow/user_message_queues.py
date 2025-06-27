from queue import Queue, Empty
from threading import Lock

from fastworkflow.utils.logging import logger

class UserMessageQueues:
    def __init__(self):
        self._queues: dict[int, Queue] = {}
        self._lock = Lock()
   
    def get_queue(self, workflow_id: int) -> Queue:
        with self._lock:
            return self._queues[workflow_id]
 
    def add_queue(self, workflow_id: int) -> None:
        with self._lock:
            self._queues[workflow_id] = Queue()
    
    def remove_queue(self, workflow_id: int) -> None:
        """Remove a workflow's queue after draining"""
        with self._lock:
            if workflow_id not in self._queues:
                return

            if remaining := self.drain_queue(workflow_id):
                logger.warning(f"Removed queue for workflow {workflow_id} with {len(remaining)} pending messages")

            del self._queues[workflow_id]     
        
    def drain_queue(self, workflow_id: int) -> list[str]:
        """Drain all messages from a queue before removal"""
        with self._lock:
            if workflow_id not in self._queues:
                return []
            
            messages = []
            queue = self._queues[workflow_id]
            while not queue.empty():
                try:
                    messages.append(queue.get_nowait())
                except Empty:
                    break
            return messages
