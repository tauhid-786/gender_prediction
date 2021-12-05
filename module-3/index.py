from flask import Flask,render_template,redirect
# __name__==__main__
friends=["tauhid","amir","aatif","chirag"]
app = Flask(__name__)
@app.route('/')
def hello():
    return render_template("index.html",my_friends=friends)
@app.route('/home')
def home():
    return redirect('/')
@app.route('/about')
def about():
    return "<h1> about page </h1>"
if __name__ == '__main__':
    app.run(debug=True)
