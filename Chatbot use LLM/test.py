import google.generativeai as genai
import os

genai.configure(api_key=os.environ["AIzaSyC5XU20Se3stD5BQ7fM4kItqyj7nUxGyv0"])


model = genai.GenerativeModel("gemini-1.5-flash")
response = model.generate_content("Write a story about a magic backpack.")
print(response.text)