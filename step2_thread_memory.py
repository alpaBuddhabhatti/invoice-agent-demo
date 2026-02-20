"""
Step 2: Thread Memory Agent
============================
Demonstrates how to use thread-based memory to maintain conversation context.
This allows the agent to remember previous interactions and answer follow-up questions.

Key Concepts:
    - Thread: A persistent conversation context that stores message history
    - Memory: The agent can reference previous messages in the same thread
    - Multi-turn Conversation: Enables natural back-and-forth interactions

What This Does:
    - Creates an agent with conversation memory
    - Processes an initial invoice message
    - Answers follow-up questions using context from earlier messages

Enhancement Suggestions:
    1. Persist threads to database for long-term storage
    2. Add thread management (list, delete, archive threads)
    3. Implement thread sharing between users for collaboration
    4. Add conversation summarization for long threads
    5. Support multiple concurrent threads per user
    6. Implement context window management (truncate old messages)
    7. Add thread export functionality (PDF, JSON, HTML)
    8. Implement semantic search across thread history
    9. Add conversation analytics (sentiment, topics, entities)
    10. Support branching conversations (what-if scenarios)
"""

import asyncio
from agent_framework import ChatAgent
from client import get_chat_client

async def main():
    """
    Main function demonstrating thread-based conversation memory.
    
    Enhancement Ideas:
        - Load existing threads from storage
        - Add thread metadata (created_at, user_id, tags)
        - Implement thread lifecycle management
        - Add conversation quality scoring
    """
    # Create agent with instructions for answering invoice questions
    # This agent is designed for interactive Q&A about invoices
    agent = ChatAgent(
        chat_client=get_chat_client(), 
        instructions='Answer invoice questions.'
    )
    
    # Create a new thread (conversation session)
    # The thread maintains message history for context
    # Enhancement: Load thread by ID to continue previous conversations
    thread = agent.get_new_thread()
    
    # First interaction: Provide invoice information
    # This establishes context in the thread
    await agent.run('Invoice from Contoso for 1200 USD', thread=thread)
    
    # Second interaction: Ask a follow-up question
    # The agent can answer because it remembers the previous message
    result = await agent.run('What is the total amount?', thread=thread)
    
    # Display the answer
    # The agent should respond with "1200 USD" using the context from the thread
    print(result.text)
    
    # Enhancement: Save thread for future use
    # Enhancement: Add more follow-up questions to demonstrate memory

# Entry point for the script
if __name__ == '__main__':
    asyncio.run(main())
