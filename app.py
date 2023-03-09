import chat_bot
from flask import Flask, request, render_template


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('hello.html')

@app.route('/', methods=['POST'])
def form():
    input_text = request.form['input_text']
    output_text=chat_bot.chat_bot(input_text)
    return render_template('hello.html', input_text=output_text)

if __name__ == '__main__':
    app.run(debug=True)
