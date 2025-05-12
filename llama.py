# ollama_chat.py

import ollama
from memorymanager import MemoryManager


MODEL_NAME = "qwen3:14b"


def simple_summarizer(messages):
    # Extract key information
    user_messages = [m for m in messages if m["role"] == "user"]
    important_points = [
        m for m in messages if m["role"] == "system" and "Summary" not in m["content"]
    ]

    texts = [m["content"] for m in important_points] + [
        f"User: {m['content']}" for m in user_messages
    ]

    joined_text = "\n".join(texts)
    prompt = f"Summarize this conversation while preserving key facts, pay attention to messages by me the user and any information I gave you like my personal information:\n\n{joined_text}"

    response = ollama.chat(
        model=MODEL_NAME, messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"].strip()


memory_manager = MemoryManager(
    path="memory.json", max_messages=50, summarizer=simple_summarizer
)


def summarize_memory_tool():
    # Get current memory
    memory = memory_manager.get_memory()

    # Get the latest summary if it exists
    latest_summary = None
    for message in reversed(memory):
        if (
            message["role"] == "system"
            and "Summary of previous conversation:" in message["content"]
        ):
            latest_summary = message["content"]
            break

    # First summarize the memory
    memory_manager.summarize_memory()

    # Create a response that includes reading the summary
    if latest_summary:
        return f"""I've reviewed the conversation history. Here's what I found:
{latest_summary}

I'll keep these points in mind as we continue our discussion."""
    else:
        return (
            "Memory has been summarized, but no significant previous context was found."
        )


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


def read_full_memory_tool():
    memory = memory_manager.get_memory()
    formatted_memory = []

    for msg in memory:
        if msg["role"] == "assistant" and "note" in msg:
            formatted_memory.append(f"📝 Saved Note: {msg['content']}")
        elif msg["role"] == "system" and "Summary" in msg["content"]:
            formatted_memory.append(f"🔄 {msg['content']}")
        elif msg["role"] == "user":
            formatted_memory.append(f"User said: {msg['content']}")

    return "\n\n".join(formatted_memory)


tools = [
    {
        "name": "summarize_memory",
        "description": "Summarizes the current chat memory and reads it back to understand context. Use this when you need to refresh your understanding of the conversation history or check for specific details from earlier.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "save_note_to_memory",
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
        "name": "revise_message",
        "description": "Revise a previous assistant message if an error was made. Also lets you read your previous messages and make any corrections to your answers from here on.",
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
    {
        "name": "read_full_memory",
        "description": "Read the entire conversation history including saved notes and summaries. Use this when you need to check for specific information that was previously stored.",
        "parameters": {"type": "object", "properties": {}},
    },
]

system_prompt = """
You are an AI assistant with access to the following memory management tools:

1. summarize_memory
   - Use when: Conversation is long or you need to refresh context
   - Function: Summarizes chat history and reads it back to you
   - No parameters needed
   - Returns: A summary of the conversation

2. revise_message
   - Use when: You need to correct a factual error in your previous responses
   - Parameters:
     * message_index: The index of the message to revise
     * new_content: The corrected content
   - Returns: Confirmation of revision

3. save_note_to_memory
   - Use when: You need to store important information for future reference
   - Parameters:
     * note: The content to remember
   - Returns: Confirmation of save

4. read_full_memory
   - Use when: You need to review the entire conversation history
   - Function: Reads all stored messages and summaries
   - No parameters needed
   - Returns: Complete conversation history with formatting

Guidelines:
- You ALWAYS have access to these tools - they are fully implemented and ready to use
- Use tools proactively when needed
- Acknowledge when you use memory tools
- Maintain consistency with previously stored information
- When using tools, format them exactly as:
  {"name": "tool_name", "arguments": {"param": "value"}}

Examples:
- To summarize: {"name": "summarize_memory"}
- To save note: {"name": "save_note_to_memory", "arguments": {"note": "Important fact here"}}
- To revise: {"name": "revise_message", "arguments": {"message_index": 1, "new_content": "Corrected content"}}
- To read all: {"name": "read_full_memory"}

Important:
- ALWAYS use read_full_memory when asked about previous information
- Use save_note_to_memory to store important user details
- Check memory before saying you don't have information
- Acknowledge when you find or don't find requested information

Example usage:
When user asks "what's my cat's name?":
1. Use read_full_memory to check stored information
2. If found, respond with the information
3. If not found, ask for the information and use save_note_to_memory to store it
"""


def ensure_system_prompt():
    memory = memory_manager.get_memory()
    if not memory:
        memory_manager.memory = [{"role": "system", "content": system_prompt}]
    else:
        # Always ensure first message is current system prompt
        if memory[0]["role"] == "system":
            memory[0]["content"] = system_prompt
        else:
            memory_manager.memory.insert(
                0, {"role": "system", "content": system_prompt}
            )
    memory_manager.save_memory()


def chat():
    ensure_system_prompt()
    print(f"Chat session started with {MODEL_NAME}. Type 'exit' to quit.\n")

    # Add tool reminder every few messages
    message_count = 0
    tool_reminder = {
        "role": "system",
        "content": "Remember: You have access to summarize_memory, revise_message, and save_note_to_memory tools.",
    }

    while True:
        # Add reminder every 10 messages
        message_count += 1
        if message_count % 10 == 0:
            memory_manager.add_message("system", tool_reminder["content"])

        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Chat Exited!")
            break

        memory_manager.add_message("user", user_input)
        conversation = memory_manager.get_memory()
        result = None
        tool_used = False

        response = ollama.chat(model=MODEL_NAME, messages=conversation, tools=tools)

        if "tool_calls" in response["message"]:
            for tool_call in response["message"]["tool_calls"]:
                tool_name = tool_call["function"]["name"]
                tool_used = True

                if tool_name == "summarize_memory":
                    print("Assistant requested to summarize memory!")
                    result = summarize_memory_tool()

                elif tool_name == "revise_message":
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

                elif tool_name == "save_note_to_memory":
                    print("Assistant requested to save a note to memory!")
                    arguments = tool_call["function"].get("arguments", {})
                    note = arguments.get("note")

                    if note:
                        result = save_note_to_memory_tool(note)

                elif tool_name == "read_full_memory":
                    print("Assistant requested to read full memory!")
                    result = read_full_memory_tool()

                else:
                    result = f"Unknown tool called: {tool_name}"

                if result:
                    memory_manager.add_message("tool", result)

            if tool_used:
                continue

        assistant_message = response["message"]["content"]
        print(f"\nAssistant: {assistant_message}\n")
        memory_manager.add_message("assistant", assistant_message)


if __name__ == "__main__":
    chat()
