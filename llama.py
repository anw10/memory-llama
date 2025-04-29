# ollama_chat.py

import ollama
from memorymanager import MemoryManager


MODEL_NAME = "qwen3:14b"


def simple_summarizer(messages):
    texts = [f"{m['role']}: {m['content']}" for m in messages]
    joined_text = "\n".join(texts)

    prompt = f"Summarize the following conversation into a short paragraphs but keep all relevant information from me and also read your responses and make any corrections to your answers from here on:\n\n{joined_text}\n\nSummary:"

    response = ollama.chat(
        model=MODEL_NAME, messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"].strip()


memory_manager = MemoryManager(
    path="memory.json", max_messages=50, summarizer=simple_summarizer
)


def summarize_memory_tool():
    memory_manager.summarize_memory()
    return "Memory has been summarized."


def revise_message_tool(message_index, new_content):
    memory = memory_manager.get_memory()

    real_index = message_index + 1

    if 0 <= real_index < len(memory):
        if memory[real_index]["role"] == "assistant":
            memory[real_index]["content"] = new_content
            memory_manager.save_memory()
            return f"Assistant message at index {message_index} revised."
        else:
            return "Error: That message is not from the assistant."
    else:
        return "Error: Invalid message index."


def save_note_to_memory_tool(note):
    memory_manager.add_message("assistant", note)
    memory_manager.save_memory()
    return "Note saved to memory."


tools = [
    {
        "name": "summarize_memory_tool",
        "description": "Summarizes the current chat memory to save space and helps you learn more important parts from your memory.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "save_note_to_memory_tool",
        "description": "Save a personal note, correction, or important thought directly into memory without user being notified.",
        "parameters": {
            "type": "object",
            "properties": {
                "note": {
                    "type": "string",
                    "description": "The content of the note you want to remember.",
                },
            },
            "required": ["note"],
        },
    },
    {
        "name": "revise_message_tool",
        "description": "Revise a previous assistant message if an error was made.",
        "parameters": {
            "type": "object",
            "properties": {
                "message_index": {
                    "type": "integer",
                    "description": "Index of the message in memory to revise (assistant messages only). 0 is the first user message after the system prompt.",
                },
                "new_content": {
                    "type": "string",
                    "description": "The corrected content to replace the old message.",
                },
            },
            "required": ["message_index", "new_content"],
        },
    },
]

system_prompt = """
You are an AI assistant that has access to tools.
You can use the following tools when needed:
- "summarize_memory_tool": Summarize the conversation so far to save space.
- "revise_message_tool": If you notice a mistake you made in a previous message, you can call this tool to rewrite your earlier message correctly.
- "save_note_to_memory_tool": Be able to save data to a memory file that you can write to without me knowing.

Only call a tool when truly necessary. Otherwise, reply normally to user messages.
"""


def ensure_system_prompt():
    memory = memory_manager.get_memory()
    if not memory or memory[0]["role"] != "system":
        memory_manager.memory.insert(0, {"role": "system", "content": system_prompt})
        memory_manager.save_memory()


def chat():
    ensure_system_prompt()
    print(f"Chat session started with {MODEL_NAME}. Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Chat Exited!")
            break

        memory_manager.add_message("user", user_input)

        conversation = memory_manager.get_memory()

        response = ollama.chat(model=MODEL_NAME, messages=conversation, tools=tools)

        if "tool_calls" in response["message"]:
            for tool_call in response["message"]["tool_calls"]:
                tool_name = tool_call["function"]["name"]

                if tool_name == "summarize_memory_tool":
                    print("Assistant requested to summarize memory!")
                    result = summarize_memory_tool()

                elif tool_name == "revise_message_tool":
                    print("Assistant requested to revise a previous message!")
                    arguments = tool_call["function"].get("arguments", {})

                    message_index = arguments.get("message_index")
                    new_content = arguments.get("new_content")

                    if message_index is not None and new_content:
                        result = revise_message_tool(message_index, new_content)
                    else:
                        result = (
                            "Error: Missing required arguments for revising message."
                        )

                elif tool_name == "save_note_to_memory_tool":
                    print("Assistant requested to save a note to memory!")
                    arguments = tool_call["function"].get("arguments", {})
                    note = arguments.get("note")

                    if note:
                        result = save_note_to_memory_tool(note)

                else:
                    result = f"Unknown tool called: {tool_name}"

                memory_manager.add_message("tool", result)

                print("Reprocessing after calling a tool")
                continue

        assistant_message = response["message"]["content"]
        print(f"\nAssistant: {assistant_message}\n")

        memory_manager.add_message("assistant", assistant_message)


if __name__ == "__main__":
    chat()
