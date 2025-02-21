import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set the FastAPI backend URL
API_URL = "http://localhost:8000/chat/"

st.title("🚢 Titanic AI Chatbot")
st.markdown("Ask anything about the Titanic dataset!")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input field
query = st.text_input("Your question:")

# Chart type selection (optional)
chart_type = st.selectbox(
    "Select chart type (optional)", 
    ["None", "histogram", "bar", "scatter", "line"], 
    index=0
)

if st.button("Ask"):
    if query:
        # Prepare request payload
        payload = {
            "user_query": query,
            "chart_type": chart_type if chart_type != "None" else ""
        }

        try:
            # Send request to FastAPI
            response = requests.post(API_URL, json=payload).json()

            # ✅ Check if "response" exists before accessing
            if "response" in response:
                bot_reply = response["response"]
            else:
                bot_reply = "⚠️ No response received from the API."

            # Store chat history
            st.session_state.chat_history.append(("You", query))
            st.session_state.chat_history.append(("Bot", bot_reply))

            # Display chat history
            st.write("### Chat History")
            for role, text in st.session_state.chat_history:
                st.write(f"**{role}:** {text}")

            # ✅ Handle Chart Data Dynamically
            chart_data = response.get("chart_data")
            if chart_data:
                st.write("### Visualization")
                fig, ax = plt.subplots()

                chart_type = chart_data.get("type", "bar")  # Default to bar

                if chart_type == "histogram":
                    sns.histplot(chart_data["data"], bins=20, kde=True, ax=ax)
                
                elif chart_type == "bar":
                    data_df = pd.DataFrame(list(chart_data["data"].items()), columns=["Category", "Count"])
                    st.bar_chart(data_df.set_index("Category"))

                st.pyplot(fig)
            else:
                st.write("⚠️ No chart data received.")

        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.write("Raw API Response:", response)  # Debugging info
