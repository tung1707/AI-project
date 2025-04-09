from flask import Flask, request, render_template,url_for,request
from transformers import pipeline
from openai import OpenAI

app = Flask(__name__)
client = OpenAI(
   api_key = ''
)

# AIzaSyC5XU20Se3stD5BQ7fM4kItqyj7nUxGyv0

# def get_completion(prompt): 
#     query = client.completions.create( 
#         model="gpt-3.5-turbo-instruct", 
#         prompt=prompt, 
#         max_tokens=20, 
#         n=1, 
#         stop=None, 
#         temperature=0.5, 
#     ) 
  
#     response = query.choices[0].text 
#     return response 

@app.route("/chatbot",methods=['GET','POST'])
def response():
    response = ""
    #facebook opt 350m
    generator = pipeline('text-generation',model = "facebook/opt-350m")
    if request.method == 'POST':
        question = request.form['question']
        response =  generator(question) #facebook 350m
        # response =  get_completion(question) #chatgpt    
    return render_template("chatbot.html",response = response)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/')
def home():
    return render_template("home.html")

if __name__ == "__main__":
    app.run(debug=True)