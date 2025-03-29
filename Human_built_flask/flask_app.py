from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
import os
import subprocess
import json
from werkzeug.utils import secure_filename
import uuid
from query_data import query_rag

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Change this in production

# Configuration
UPLOAD_FOLDER = 'data'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx', 'csv', 'json', 'xml'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    # Get list of files in data directory
    data_files = []
    if os.path.exists(UPLOAD_FOLDER):
        data_files = [f for f in os.listdir(UPLOAD_FOLDER) if os.path.isfile(os.path.join(UPLOAD_FOLDER, f))]
    
    return render_template('index.html', data_files=data_files)

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    files = request.files.getlist('file')
    
    # If user does not select file, browser submits an empty file
    if not files or files[0].filename == '':
        flash('No selected file')
        return redirect(url_for('index'))
    
    # Create upload folder if it doesn't exist
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    
    flash('Files uploaded successfully')
    return redirect(url_for('index'))

@app.route('/populate_database', methods=['POST'])
def populate_database():
    try:
        # Run the populate_database.py script
        result = subprocess.run(['python', 'populate_database.py'], 
                                capture_output=True, text=True, check=True)
        flash('Database populated successfully')
        return jsonify({
            'success': True,
            'message': 'Database populated successfully',
            'output': result.stdout
        })
    except subprocess.CalledProcessError as e:
        flash('Error populating database')
        return jsonify({
            'success': False,
            'message': 'Error populating database',
            'error': e.stderr
        }), 500

@app.route('/query', methods=['POST'])
def query_database():
    query = request.form.get('query', '')
    if not query:
        flash('Query cannot be empty')
        return redirect(url_for('index'))
    
    try:
        # Run the query_data.py script with the query
        result = subprocess.run(['python', 'query_data.py', query], 
                                capture_output=True, text=True, check=True)
        
        # Try to parse the output as JSON
        try:
            output_data = json.loads(result.stdout)
        except json.JSONDecodeError:
            output_data = result.stdout
        
        return jsonify({
            'success': True,
            'query': query,
            'results': output_data
        })
    except subprocess.CalledProcessError as e:
        return jsonify({
            'success': False,
            'query': query,
            'error': e.stderr
        }), 500

# Add chat session storage
chat_histories = {}

@app.route('/chat')
def chat():
    """Interactive chatbot interface for document queries."""
    # Create a unique session ID if not present
    if 'chat_id' not in session:
        session['chat_id'] = str(uuid.uuid4())
    
    # Initialize chat history for this session if needed
    if session['chat_id'] not in chat_histories:
        chat_histories[session['chat_id']] = []
    
    return render_template('chat.html')

@app.route('/chat_query', methods=['POST'])
def chat_query():
    """Handle chat messages for document database."""
    # Get the message from request
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({
            'success': False,
            'error': 'No message provided'
        }), 400
    
    user_message = data['message']
    
    # Get or create chat session
    if 'chat_id' not in session:
        session['chat_id'] = str(uuid.uuid4())
    
    if session['chat_id'] not in chat_histories:
        chat_histories[session['chat_id']] = []
    
    # Add user message to history
    chat_histories[session['chat_id']].append({
        'role': 'user',
        'content': user_message
    })
    
    try:
        # Use the existing query_rag function to get a response
        response = query_rag(user_message)
        
        # Extract sources if available (returned by query_rag)
        sources = []
        if isinstance(response, tuple) and len(response) > 1:
            response_text, sources = response
        else:
            response_text = response
        
        # Add system response to history
        chat_histories[session['chat_id']].append({
            'role': 'assistant',
            'content': response_text
        })
        
        # Limit history length to prevent memory issues
        if len(chat_histories[session['chat_id']]) > 20:
            chat_histories[session['chat_id']] = chat_histories[session['chat_id']][-20:]
        
        return jsonify({
            'success': True,
            'response': response_text,
            'sources': sources
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)