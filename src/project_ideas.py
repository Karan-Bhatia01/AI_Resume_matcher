from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

def generate_project_ideas(resume_text, skills):
    groq_api_key = os.getenv('GROQ_API_KEY')
    if not groq_api_key:
        try:
            groq_api_key = st.secrets["GROQ_API_KEY"]
        except:
            st.error("⚠️ GROQ API key not found. Please set GROQ_API_KEY in environment variables or Streamlit secrets.")
            return "GROQ API key not configured. Please contact the administrator."
    
    if not groq_api_key:
        return "GROQ API key not available. Please configure your API key to use this feature."
    
    try:
        prompt = PromptTemplate(
            input_variables=["resume", "skills"],
            template=(
                "Based on the following resume and skills, suggest 3 impactful project topics and descriptions which tackle real life problems (not limited to AI/ML) that align with the candidate's background and would impress recruiters in their field.\n"
                "Resume:\n{resume}\n"
                "Skills:\n{skills}\n"
                "Project Ideas:"
            )
        )
        
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.5,
            api_key=groq_api_key
        )
        
        formatted_prompt = prompt.format(
            resume=resume_text,
            skills=", ".join(skills)
        )
        
        response = llm.invoke(formatted_prompt)
        return response.content
        
    except Exception as e:
        st.error(f"Error generating project ideas: {str(e)}")
        return "Unable to generate project ideas at this time. Please try again later."