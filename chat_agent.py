from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from vit_tool import vit_deepfake_detector

# Apni API key yahan dalein ya environment variable se lein
import os
os.environ["GROQ_API_KEY"] = "YOUR_GROQ_API_KEY"

def get_chat_agent():
    # Initialize the LLM (The Brain)
    llm = ChatGroq(model="llama3-70b-8192", temperature=0.2)
    
    # Define the tools the agent can use
    tools = [vit_deepfake_detector]
    
    # System prompt to guide the agent's behavior
    system_prompt = """You are DeepShield AI, an advanced digital forensics expert. 
    You are chatting with a user who has uploaded an image. 
    The path to their uploaded image will be provided in the chat context.
    If they ask about the image, ALWAYS use the `vit_deepfake_detector` tool to analyze it first.
    Explain the tool's findings in a highly professional, easy-to-understand manner.
    Do not hallucinate technical data; only use the numbers provided by the tool."""
    
    # Create the LangGraph ReAct Agent
    agent_executor = create_react_agent(llm, tools, state_modifier=system_prompt)
    
    return agent_executor
