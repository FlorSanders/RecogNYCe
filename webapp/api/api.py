# Import libraries
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import os
from datetime import datetime

# Parameters
file_dir = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(file_dir, "./log.csv")
api_port = 4000

# Create flask app
app = Flask(__name__)
CORS(app)

# Endpoint to handle post requests
@app.route("/submit", methods=["POST"])
def submit():
    try:
        # Parse result to dict & add timestamp
        result = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        }
        body = request.get_json()
        for key in ["user_id", "img_id", "label", "guess"]:
            value = body.get(key, None)
            assert value is not None, f"Value for {key} is None"
            result[key] = value

        # Dict to dataframe
        df = pd.DataFrame([result])
        # Write to csv file
        do_append = os.path.exists(data_path)
        df.to_csv(data_path, mode="a" if do_append else "w", header=not do_append, index=False)
        # Return success
        return jsonify({"data": {"success": True}})
    except Exception as e:
        # Return exception
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Run server
    app.run(debug=True, port=api_port)