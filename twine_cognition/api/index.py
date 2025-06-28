from flask import Flask, request, jsonify
import torch
import sys
import os

# Ensure local imports work on Vercel
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathos_core import PathosMatrixOrchestrator
from lagos_marix import LogosMatrixOrchestrator

app = Flask(__name__)

def run_twine_cognition(sensory_data):
    logos = LogosMatrixOrchestrator()
    pathos = PathosMatrixOrchestrator(sensory_input_size=sensory_data.shape[1])
    intuitive_markers = pathos.run_intuitive_analysis(sensory_data)
    _ = logos.run_logical_analysis(sensory_data)
    final_plan = logos.adjust_plan_with_pathos_guidance(intuitive_markers)
    return final_plan

@app.route("/api", methods=["POST"])
def api_root():
    data = request.json.get("sensory_data", None)
    if not data or len(data) != 10:
        return jsonify({"error": "Provide 10 values as 'sensory_data'"}), 400
    sensory_data = torch.tensor([data])
    result = run_twine_cognition(sensory_data)
    return jsonify({"decision": result})

# For Vercel: expose 'app' as 'vc_app'
vc_app = app