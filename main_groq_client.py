### this has nothing to do with the RAG, it is just example code for groq client usage

from groq_client import GroqClient

def example_simple_chat():
    """Example: Simple single message chat"""
    print("=== Simple Chat Example ===")
    
    client = GroqClient()
    
    response = client.simple_chat(
        user_message="Explain quantum computing in one sentence.",
        system_message="You are a helpful AI assistant.",
        temperature=0.7,
        max_tokens=100
    )
    
    print(f"Response: {response}\n")

def example_multi_turn_chat():
    """Example: Multi-turn conversation"""
    print("=== Multi-turn Chat Example ===")
    
    client = GroqClient()
    
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What is Python?"},
    ]
    
    response = client.chat(messages, temperature=0.7, max_tokens=150)
    assistant_message = response.choices[0].message.content
    
    print(f"User: What is Python?")
    print(f"Assistant: {assistant_message}\n")
    
    # Continue the conversation
    messages.append({"role": "assistant", "content": assistant_message})
    messages.append({"role": "user", "content": "What are its main features?"})
    
    response = client.chat(messages, temperature=0.7, max_tokens=200)
    print(f"User: What are its main features?")
    print(f"Assistant: {response.choices[0].message.content}\n")

def example_streaming_chat():
    """Example: Streaming response"""
    print("=== Streaming Chat Example ===")
    
    client = GroqClient()
    
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Write a short poem about coding."}
    ]
    
    print("Assistant: ", end="", flush=True)
    for chunk in client.stream_chat(messages, temperature=0.8, max_tokens=200):
        print(chunk, end="", flush=True)
    print("\n")

def example_with_custom_model():
    """Example: Using a different model"""
    print("=== Custom Model Example ===")
    
    # You can specify a different model
    client = GroqClient(model="mixtral-8x7b-32768")
    
    response = client.simple_chat(
        user_message="What model are you?",
        temperature=0.5
    )
    
    print(f"Response: {response}\n")

def main():
    """Run all examples"""
    try:
        print("Starting Groq API Examples...\n")
        
        example_simple_chat()
        example_multi_turn_chat()
        example_streaming_chat()
        # example_with_custom_model()  # Uncomment to try different model
        
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()