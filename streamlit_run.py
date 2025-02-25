import streamlit as st
import os
from dotenv import load_dotenv, find_dotenv
from code_analysis_agent import CodeAnalysisAgent

# Load environment variables
load_dotenv(find_dotenv(), override=True)

# Configure the Streamlit page
st.set_page_config(
    page_title="Neo4j Code Analysis Assistant",
    page_icon="🔍",
    layout="wide"
)

# Add custom CSS for better styling
st.markdown("""
    <style>
    .stTextInput > div > div > input {
        font-size: 16px;
    }
    .big-font {
        font-size: 20px !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize the agent (do this only once)
@st.cache_resource
def get_agent():
    """Initialize and cache the CodeAnalysisAgent."""
    NEO4J_URI = os.environ.get("NEO4J_URI")
    NEO4J_USER = os.environ.get("NEO4J_USERNAME")
    NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    
    return CodeAnalysisAgent(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, OPENAI_API_KEY)

# Get the agent
agent = get_agent()

# Header
st.title("🔍 Neo4j Code Analysis Assistant")
st.markdown("""
    Ask questions about your codebase and get detailed insights from the Neo4j knowledge graph.
    Type your question below and click 'Analyze' or press Enter.
""")

# Create a form for input
with st.form(key='query_form'):
    query = st.text_area(
        "Enter your code-related question:",
        height=100,
        placeholder="Example: How is error handling implemented in the codebase?"
    )
    
    submit_button = st.form_submit_button(label='Analyze')

# Handle form submission
if submit_button and query:
    try:
        with st.spinner('Analyzing your question...'):
            # Show the retrieval phase
            retrieval_info = st.info("🔍 Retrieving code information...")
            
            # Execute the analysis
            answer = agent.analyze(query)
            
            # Update the info message based on results
            if "not found in the codebase" in answer or "No relevant information was found" in answer:
                retrieval_info.warning("⚠️ Limited or no information found in the codebase")
            else:
                retrieval_info.success("✅ Information retrieved successfully")
        
        # Display results
        st.markdown("### Analysis Results")
        st.markdown(answer)
        
        # Add suggestions if no results found
        if "not found in the codebase" in answer or "No relevant information was found" in answer:
            st.info("""
            **Suggestions:**
            - Try different search terms or function/class names
            - Check for typos in function or class names
            - Use more general terms to find related components
            - Ask about specific files instead of functions
            """)
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please try rephrasing your question or try again later.")

# Add information in sidebar
with st.sidebar:
    st.markdown("### About")
    st.markdown("""
        This tool helps you analyze and understand your codebase using a 
        Neo4j knowledge graph-based approach. You can ask questions about:
        
        - Code structure and organization
        - Function implementations and relationships
        - Class hierarchies and dependencies
        - Error handling patterns
        - Control flow and execution paths
    """)
    
    st.markdown("### Example Questions")
    st.markdown("""
        - How is the message routing system implemented?
        - What are the main classes and their relationships?
        - How does error handling work in the data processing module?
        - What components depend on the authentication service?
        - What functions are defined in the retriever.py file?
        - How does the LLMRetriever class work?
    """)

st.markdown("---")
st.markdown(
    "Made with ❤️ using Streamlit, LangChain, LangGraph, and Neo4j",
    help="This application uses a Neo4j knowledge graph to analyze code structure and relationships."
)