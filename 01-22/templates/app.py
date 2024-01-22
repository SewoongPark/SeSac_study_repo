from flask import Flask, render_template, request
app = Flask(__name__)

@app.route('/')
def index():

    if request.method == 'GET':
        return render_template('index.html')
    
    if request.method == 'POST':
        price = float(request.form['pan1']) + float(request.form['pan2'])
        return render_template('index.html', price = price)
if __name__ == '__main__':
    app.run(debug=True)