from flask import Flask, jsonify, request, Response
import requests
import json

app = Flask(__name__)

CONFIG_FILE = "nodes.json"

# Load configuration
with open(CONFIG_FILE, 'r') as config_file:
    config = json.load(config_file)

# Function to fetch data from a given node
def fetch_data_from_node(node_url, endpoint):
    try:
        response = requests.get(f"{node_url}/{endpoint}")
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        print(f"Failed to fetch data from {node_url}: {e}")
        return None

# Function to aggregate data from all nodes
def aggregate_data():
    gpu_info_list = []
    ollama_info_list = []

    for node in config["nodes"]:
        node_gpu_info = fetch_data_from_node(node["url"], "gpu-info")
        node_ollama_info = fetch_data_from_node(node["url"], "ollama-info")
        
        if node_gpu_info:
            gpu_info_list.append({"node": node["url"], "gpu_info": node_gpu_info})
        if node_ollama_info:
            ollama_info_list.append({"node": node["url"], "ollama_info": node_ollama_info})

    return gpu_info_list, ollama_info_list

# Function to select the best Ollama instance
def select_best_ollama_instance(model_name, ollama_info_list, gpu_info_list):
    candidates = []

    # Find instances supporting the requested model
    for ollama_info in ollama_info_list:
        for service in ollama_info["ollama_info"]["ollama_services"]:
            if any(model["name"] == model_name for model in service["models_avail"]):
                candidates.append({
                    "node": ollama_info["node"],
                    "service": service,
                    "gpu_indices": service["gpu_indices"]
                })

    # Select the best instance based on GPU load and memory
    best_instance = None
    best_score = float('inf')

    for candidate in candidates:
        gpu_utilization = 0
        gpu_memory_available = 0

        for gpu_info in gpu_info_list:
            if gpu_info["node"] == candidate["node"]:
                for gpu in gpu_info["gpu_info"]["gpus"]:
                    if gpu["index"] in candidate["gpu_indices"]:
                        gpu_utilization += gpu["utilization"]
                        gpu_memory_available += int(gpu["memory_total"].split()[0]) - int(gpu["memory_used"].split()[0])

        score = gpu_utilization - gpu_memory_available
        if score < best_score:
            best_score = score
            best_instance = candidate

    return best_instance

# Function to get all available models from all nodes
def get_all_models():
    model_names = set()
    for node in config["nodes"]:
        node_ollama_info = fetch_data_from_node(node["url"], "ollama-info")
        if node_ollama_info:
            for service in node_ollama_info["ollama_services"]:
                for model in service["models_avail"]:
                    model_names.add(model["name"])
    return sorted(list(model_names))

# Helper function to pretty print JSON
def pretty_json(data):
    return json.dumps(data, indent=4)

# Load balancer endpoint
@app.route('/load-balance', methods=['POST'])
def load_balance():
    request_data = request.json
    model_name = request_data["model"]
    prompt = request_data["prompt"]
    other_params = request_data.get("other_params", {})

    gpu_info_list, ollama_info_list = aggregate_data()
    best_instance = select_best_ollama_instance(model_name, ollama_info_list, gpu_info_list)

    if best_instance:
        service_url = best_instance["service"]["url"]
        forward_url = f"{service_url}/inference"
        
        try:
            response = requests.post(forward_url, json={
                "model_name": model_name,
                "prompt": prompt,
                **other_params
            })
            return Response(pretty_json(response.json()), status=response.status_code, mimetype='application/json')
        except Exception as e:
            return Response(pretty_json({"error": str(e)}), status=500, mimetype='application/json')
    else:
        return Response(pretty_json({"error": "No suitable Ollama instance found"}), status=500, mimetype='application/json')

# Endpoint to list all available models across all nodes
@app.route('/list-models', methods=['GET'])
def list_models():
    try:
        all_models = get_all_models()
        return Response(pretty_json({"models": all_models}), mimetype='application/json')
    except Exception as e:
        return Response(pretty_json({"error": str(e)}), status=500, mimetype='application/json')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8009)

