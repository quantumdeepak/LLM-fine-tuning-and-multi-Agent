import ollama

# Generate a response using the Gemma model
response = ollama.chat(model="gemma3:4b", messages=[{"role": "user", "content": "What is machine learning? explain in 2 lines"}])

# Print the response
print(response["message"]["content"])