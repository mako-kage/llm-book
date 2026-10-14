from google import genai

client = genai.Client()

interaction = client.interactions.create(
    model="gemini-3.8-flash",
    input="who are you?"
)

res = interaction.output_text
print(res)
