from together import Together

# Set your Together.ai API key

client = Together(api_key="tgp_v1_2f3bY_-WWTP9Bo9NPSFZk1WYEFhz9aXC5m9FObxVI1o")

response = client.chat.completions.create(
    model="meta-llama/Llama-3-8b-chat-hf",
    messages=[
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Explain the purpose of a Graph Neural Network in simple terms."}
    ],
    temperature=0.7,
    max_tokens=300,
)

print(response.choices[0].message.content.strip())
