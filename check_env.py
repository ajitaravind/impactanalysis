import os
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv(), override=True)

# Check environment variables
print("Environment Variables Check:")
print("-" * 50)

# Check OpenAI API key
openai_api_key = os.environ.get("OPENAI_API_KEY")
if openai_api_key:
    print(f"OPENAI_API_KEY: {'*' * 5}{openai_api_key[-4:]} (length: {len(openai_api_key)})")
else:
    print("OPENAI_API_KEY: Not set")

# Check Pinecone API key
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
if pinecone_api_key:
    print(f"PINECONE_API_KEY: {'*' * 5}{pinecone_api_key[-4:]} (length: {len(pinecone_api_key)})")
else:
    print("PINECONE_API_KEY: Not set")

# Check Pinecone environment
pinecone_env = os.environ.get("PINECONE_ENVIRONMENT")
print(f"PINECONE_ENVIRONMENT: {pinecone_env}")

# Check Neo4j variables
neo4j_uri = os.environ.get("NEO4J_URI")
neo4j_user = os.environ.get("NEO4J_USERNAME")
neo4j_password = os.environ.get("NEO4J_PASSWORD")

print(f"NEO4J_URI: {neo4j_uri}")
print(f"NEO4J_USERNAME: {neo4j_user}")
if neo4j_password:
    print(f"NEO4J_PASSWORD: {'*' * len(neo4j_password)}")
else:
    print("NEO4J_PASSWORD: Not set")

print("\nRecommendations:")
print("-" * 50)

# Check for potential issues
if not pinecone_api_key:
    print("- Set PINECONE_API_KEY in your .env file")
    
if not pinecone_env:
    print("- Set PINECONE_ENVIRONMENT in your .env file (e.g., 'us-east-1')")
    
if not openai_api_key:
    print("- Set OPENAI_API_KEY in your .env file")
    
if not all([neo4j_uri, neo4j_user, neo4j_password]):
    print("- Set NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD in your .env file")

# Check for potential mix-ups
if pinecone_api_key and pinecone_api_key.startswith("sk-"):
    print("- WARNING: Your PINECONE_API_KEY looks like an OpenAI API key. Please check your .env file.")
    
if openai_api_key and not openai_api_key.startswith("sk-"):
    print("- WARNING: Your OPENAI_API_KEY doesn't start with 'sk-'. Please check your .env file.")

print("\nTo fix these issues, update your .env file with the correct values.") 