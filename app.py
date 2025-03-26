from flask import Flask, jsonify, request
import subprocess

app = Flask(__name__, static_folder='.', static_url_path='')

@app.route('/')
def index():
    return app.send_static_file('demo.html')

@app.route('/generate', methods=['POST'])
def generate():
    try:
        # Extract values from incoming JSON request
        data = request.json
        foreground = data.get("foreground")
        background = data.get("background")
        N = data.get("frames")
        
        # Run demo.py with the extracted values as args
        result = subprocess.run(
            ["python3", "demo.py", N, foreground, background],
            capture_output=True,
            text=True
        )

        # Capture both stdout and stderr
        output = result.stdout
        error = result.stderr

        # Return the output and error as part of the response -> remove if we don't want annoying pop up at end
        if result.returncode == 0:
            return jsonify({"output": output})
        else:
            return jsonify({"error": error}), 500
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
