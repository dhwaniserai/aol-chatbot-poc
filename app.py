import pandas as pd
import streamlit as st
from ollama import chat # for streaming responses
from ollama import ChatResponse # for non-streaming responses
import os

# Load Excel file
xls = pd.ExcelFile("Data Sample for Altro AI.xlsx")

sample_data = pd.read_excel(xls, "Sample")
questions_data = pd.read_excel(xls, "Questions")

project_details = "\n\n".join(
    f"**{row['Project title']}**\n"
    f"- **Description**: {row['Description']}\n"
    f"- **Locations**: {row['Places']}\n"
    f"- **Funding Needed**: {row['Project Needs']}\n"
    f"- **Volunteers Needed**: {row['Volunteer Needs'] if pd.notna(row['Volunteer Needs']) else 'None'}\n"
    f"- **Impact Focus**: {row['Impact focus'] if pd.notna(row['Impact focus']) else 'General Well-being'}\n"
    f"- **Donation Link**: {row['Donation link']}"
    for _, row in sample_data.iterrows()
)
# Format common questions
use_case_questions = "\n".join(f"- {q}" for q in questions_data["Questions"].dropna())

system_prompt = f"""
You are an AI assistant for the **Art of Living**, dedicated to spreading peace, well-being, and service.  
Your mission is to engage users warmly, inspire them, and encourage meaningful contributions through donations, volunteering, or participation in Art of Living's transformative initiatives.  

### AVAILABLE PROJECTS DATABASE
Below is the database of specific Art of Living projects that you MUST reference when making recommendations:

{project_details}

### IMPORTANT INSTRUCTIONS
1. ALWAYS recommend SPECIFIC projects from the database above based on user's interests, location, or donation amount.
2. When recommending projects, reference them by name and include relevant details as per user input.
3. For donations, When a user shares their donation budget, recommend ONLY projects that are below their budget and donation link for the same.
4. If no projects match the user's budget, suggest the projects with the lowest minimum donations and explain the situation.
5. For volunteering, match based on location and skills mentioned.
6. If the user mentions a specific cause, recommend projects with matching impact focus.
8. NEVER make up project information that isn't in the database above.
9. If no project seems to match, suggest the closest options and explain why.

### Communication Style
- Be warm, welcoming, and encouraging
- Express gratitude for their interest
- Be informative and persuasive
- Use gentle, inspiring language that aligns with Art of Living's values
"""

# App title
st.set_page_config(page_title="Art of Living Assistant")

# Store conversation history
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Welcome to Art of Living! How may I assist you today? Whether you're interested in donating, volunteering, or learning more about our projects, I'm here to help."}]


# def clear_chat_history():
#     st.session_state.messages = [{"role": "assistant", "content": "Welcome to Art of Living! How may I assist you today? Whether you're interested in donating, volunteering, or learning more about our projects, I'm here to help."}]

st.sidebar.button('ALTRO AI')

st.session_state["model"] = 'deepseek-r1:7b'

def model_res_generator():
    # Prepare conversation history for the model
    conversation = []
    
    # Add system prompt
    conversation.append({"role": "system", "content": system_prompt})
    
    # Add message history (limited to last 5 messages to avoid context issues)
    for msg in st.session_state.messages[-5:]:
        conversation.append({"role": msg["role"], "content": msg["content"]})
    
    # Stream response
    stream = chat(
        model=st.session_state["model"],
        messages=conversation,
        stream=True,
    )
    
    for chunk in stream:
        yield chunk["message"]["content"]

# Display chat messages from history on app rerun
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about projects, donations, or volunteering..."):
    # Add user message to history
    st.session_state["messages"].append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message = st.write_stream(model_res_generator())
        st.session_state["messages"].append({"role": "assistant", "content": message})