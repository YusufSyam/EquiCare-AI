from flask import Flask, render_template, request, redirect, url_for, session, flash
from src.rag_pipeline import build_rag_pipeline  
import markdown
import os

app = Flask(__name__)
app.static_folder = 'static'
app.secret_key = os.urandom(24) 

print("Loading RAG pipeline...")
rag_chain = build_rag_pipeline()
print("RAG pipeline loaded successfully!")

def cleanse_response(old_response):
    answerForeword= "Answer:"
    answer_start = old_response.find(answerForeword) + len(answerForeword)
    new_response = old_response[answer_start:].replace("<|assistant|>", "").strip()

    return new_response

@app.route('/')
def index():
    if 'chat_history' not in session:
        session['chat_history'] = []
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
        # print(bot_response)
        
        bot_response_html = markdown.markdown(
            bot_response,
            extensions=['extra', 'nl2br'] 
        )
    except Exception as e:
        bot_response_html  = f"Error: {str(e)}. Please try again."
    
    session['chat_history'].append({'role': 'bot', 'content': bot_response_html })
    
    if len(session['chat_history']) > 20:
        session['chat_history'] = session['chat_history'][-20:]
    
    session.modified = True  # Update session
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
