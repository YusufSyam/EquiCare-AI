from flask import Flask, render_template, request, redirect, url_for, session, flash
from src.rag_pipeline import build_rag_pipeline  
import markdown
import os

app = Flask(__name__)
app.static_folder = 'static'
app.secret_key = os.urandom(24) # ini supaya session baru setiap dirun

print("Loading RAG pipeline...")
rag_chain = build_rag_pipeline()
print("RAG pipeline loaded successfully!")

def cleanse_response(old_response):
    answerForeword = "Answer:"
    answer_start = old_response.find(answerForeword) + len(answerForeword)
    new_response = old_response[answer_start:].replace("<|assistant|>", "").strip()
    return new_response

STARTER_MESSAGE_MD = """Hi! 🐴👋 I’m **EquiCare AI**, your trusted assistant for horse health and diseases.
I can help you understand symptoms, possible conditions, and preventive care for your horse.

Here are some areas you might be interested in:
1. Common Horse Diseases 🩺
2. Symptom Checker 🤒
3. Prevention & Care Tips 🌿
4. Emergency Guidelines 🚨

Feel free to ask me anything about your horse’s health!"""

def markdown_to_html(md_text):
    return markdown.markdown(
        md_text,
        extensions=['extra', 'nl2br'] 
    )

@app.route('/')
def index():
    if 'chat_history' not in session:
        session['chat_history'] = []
    
    if not session['chat_history']:
        starter_html = markdown_to_html(STARTER_MESSAGE_MD)
        session['chat_history'].append({
            'role': 'bot', 
            'content': starter_html
        })
        session.modified = True 
    
    return render_template('index.html', chat_history=session['chat_history'])

@app.route('/chat', methods=['POST'])
def chat():
    if 'chat_history' not in session:
        session['chat_history'] = []
    
    user_query = request.form.get('query')
    if not user_query.strip():
        flash('Please enter a message.')
        return redirect(url_for('index'))
    
    session['chat_history'].append({'role': 'user', 'content': user_query})
    
    try:
        result = rag_chain.invoke({"input": user_query})
        bot_response = cleanse_response(result["answer"])
        
        bot_response_html = markdown_to_html(bot_response)
    except Exception as e:
        bot_response_html = f"Error: {str(e)}. Please try again."
    
    session['chat_history'].append({'role': 'bot', 'content': bot_response_html})
    
    if len(session['chat_history']) > 20:
        session['chat_history'] = session['chat_history'][-20:]
    
    session.modified = True 
    return redirect(url_for('index'))

if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=5000)
    app.run(debug=True, host='0.0.0.0', port=5000)
