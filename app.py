from flask import Flask , render_template, request
from stock_predictor import checker, stock_res , lister

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/stock", methods = ["POST" ,"GET"])
def stock():
    if request.method == "POST":
        s = request.form["stock"]
        s = str(s).upper()
        vv  , lstname = checker(s)
        if vv == "1":
            boom = 5
            name , error , mod = stock_res(s)
            return render_template("stock.html" , name= name, error = error , stk_name = s , pred = mod ,boom = boom , lstname = lstname)
        else:
            boom = 1
            return render_template("stock.html", boom = boom)

    boom = 0
    return render_template ("stock.html" , boom = boom)


@app.route("/list")
def list():
    ids , name , industry = lister()
    return render_template("/list.html" , i = ids , n = name , indus = industry)

if __name__ == "__main__":
    app.run(host="0.0.0.0" , port = 8080)