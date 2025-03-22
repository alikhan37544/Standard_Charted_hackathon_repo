from flask import Flask, render_template

app = Flask(__name__)

# Route for the index page
@app.route('/')
def index():
    return render_template('index.html')

# Route for the loan page
@app.route('/loan')
def loan():
    return render_template('loan.html')

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5050)