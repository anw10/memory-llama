import json
import os


class MemoryManager:
    def __init__(self, path="memory.json", max_messages=50, summarizer=None):
        """
        path: file to store memory
        max_messages: number of messages before summarizing
        summarizer: a callable that can summarize a list of messages
        """
        self.path = path
        self.max_messages = max_messages
        self.summarizer = summarizer
        self.memory = self.load_memory()

    def load_memory(self):
        if os.path.exists(self.path):
            with open(self.path, "r") as f:
                return json.load(f)
        return []

    def save_memory(self):
        with open(self.path, "w") as f:
            json.dump(self.memory, f, indent=2)

    def add_message(self, role, content):
        self.memory.append({"role": role, "content": content})

        if len(self.memory) > self.max_messages:
            self.summarize_memory()

        self.save_memory()

    def get_memory(self):
        return self.memory

    def clear_memory(self):
        self.memory = []
        self.save_memory()

    def summarize_memory(self):
        if not self.summarizer:
            print("No summarizer provided, cannot summarize memory.")
            self.memory = self.memory[-self.max_messages :]
            return

        midpoint = len(self.memory) // 2
        to_summarize = self.memory[:midpoint]

        summary_text = self.summarizer(to_summarize)

        summary_message = {
            "role": "system",
            "content": f"Summary of previous conversation: {summary_text}",
        }

        self.memory = [summary_message] + self.memory[midpoint:]
