from flask import Flask, jsonify, request

import demo

app = Flask(__name__, static_folder=".", static_url_path="")


@app.route("/")
def index():
    return app.send_static_file("demo.html")


@app.route("/generate", methods=["POST"])
def generate():
    try:
        # Extract values from incoming JSON request
        data = request.json
        foreground = data.get("foreground")
        background = data.get("background")
        N = data.get("frames")

        # Call demo.py generate func
        output = demo.generate(int(N), foreground, background)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return "Done"


if __name__ == "__main__":
    app.run(debug=True)
