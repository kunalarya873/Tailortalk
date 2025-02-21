import os
import pandas as pd
import matplotlib.pyplot as plt
from pydantic import BaseModel
import seaborn as sns
from fastapi import FastAPI, HTTPException
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.memory import ConversationBufferMemory

# Load Titanic dataset
df = pd.read_csv("titanic.csv")

# Initialize FastAPI
app = FastAPI()

# Initialize Groq LLM
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.7,
    api_key=os.getenv("GROQ_API_KEY")
)

# Memory for maintaining chat history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Define expected JSON structure
parser = JsonOutputParser(pydantic_object={
    "type": "object",
    "properties": {
        "response": {"type": "string"},
        "visualization": {"type": "string", "nullable": True}
    }
})

# Define prompt for Groq
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a Titanic data expert. Answer user questions using the Titanic dataset. 
    Respond in JSON format:

    {{
        "response": "Your textual answer here",
        "visualization": "optional visualization type (e.g., 'histogram')"
    }}
    """),
    ("user", "{query}")
])


# Create LangChain pipeline
chain = prompt | llm | parser

class ChatRequest(BaseModel):
    user_query: str  # FastAPI expects this in JSON
    chart_type: str  # Added chart_type to specify which graph to return


@app.post("/chat/")
async def chat_titanic(request: ChatRequest):
    try:
        history = memory.load_memory_variables({}).get("chat_history", [])

        result = chain.invoke({"query": request.user_query, "chat_history": history})
        memory.save_context({"query": request.user_query}, {"response": result["response"]})

        query = request.user_query.lower().strip()  # Normalize user query

        chart_data = None  # Default to None

        # 🎯 Dynamically detect query intent
        if "age" in query:
            chart_data = {"type": "histogram", "data": df["Age"].dropna().tolist()}
        elif "embarked" in query or "port" in query:
            embark_counts = df["Embarked"].value_counts().to_dict()
            chart_data = {"type": "bar", "data": embark_counts}
        elif "fare" in query:
            chart_data = {"type": "histogram", "data": df["Fare"].dropna().tolist()}
        elif "class" in query:
            class_counts = df["Pclass"].value_counts().to_dict()
            chart_data = {"type": "bar", "data": class_counts}

        # 🛑 Fallback Chart (Debugging)
        if chart_data is None:
            chart_data = {"type": "histogram", "data": df["Fare"].dropna().tolist()}  # Fallback to Fare Distribution


        return {
            "response": result['response'],
            "chart_data": chart_data
        }

    except Exception as e:
        print("ERROR:", str(e), flush=True)  # Debugging
        raise HTTPException(status_code=500, detail=str(e))
