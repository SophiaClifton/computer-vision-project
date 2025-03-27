from flask import Flask, jsonify, request

import multiprocessing
import demo

app = Flask(__name__, static_folder=".", static_url_path="")


@app.route("/")
def index():
    return app.send_static_file("demo.html")

def run_generate(N, foreground, background):
    demo.generate(int(N), foreground, background)


@app.route("/generate", methods=["POST"])
def generate():
    try:
        # Extract values from incoming JSON request
        data = request.json
        foreground = data.get("foreground")
        background = data.get("background")
        N = data.get("frames")

        # Call demo.py generate in a new process so it can cleanly terminate after execution
        process = multiprocessing.Process(target=run_generate, args=(N, foreground, background))
        process.start()
        process.join()

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    return "Done"


if __name__ == "__main__":
    app.run(debug=True)
