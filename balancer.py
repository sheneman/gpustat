from flask import Flask, jsonify, request, Response, stream_with_context
from flask_cors import CORS
import requests
import json
import itertools

app = Flask(__name__)
CORS(app)

CONFIG_FILE = "nodes.json"

# Load configuration
with open(CONFIG_FILE, 'r') as config_file:
    config = json.load(config_file)

# Round-robin iterator for load balancing
def round_robin(iterable):
    pool = tuple(iterable)
    n = len(pool)
    if n == 0:
        return iter([])  # empty iterator
    cycle = itertools.cycle(pool)
    while True:
        for i in range(n):
            yield next(cycle)

# Create round-robin iterator for each model
round_robin_iterators = {}

# Function to fetch data from a given node
def fetch_data_from_node(node_url, endpoint, payload=None, method='GET', stream=False):
    try:
        if method == 'POST':
            response = requests.post(f"{node_url}/{endpoint}", json=payload, stream=stream)
        else:
            response = requests.get(f"{node_url}/{endpoint}", stream=stream)
        
        if response.status_code == 200:
            if stream:
                return response.iter_content(chunk_size=10 * 1024)  # 10 KB chunks
            return response.json()
        else:
            return None
    except Exception as e:
        print(f"Failed to fetch data from {node_url}: {e}")
        return None

# Function to check the health of a node
def check_node_health(node_url):
    try:
        response = requests.get(f"{node_url}/health")
        return response.status_code == 200
    except Exception as e:
        print(f"Failed to check health of {node_url}: {e}")
        return False

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

# Function to collect and aggregate hierarchical information
def aggregate_hierarchical_data():
    aggregated_info = []
    for node in config["nodes"]:
        node_url = node["url"]
        node_health = check_node_health(node_url)
        node_info = {
            "node": node_url,
            "health": node_health,
            "ollama_info": [],
            "gpu_info": []
        }
        ollama_info = fetch_data_from_node(node_url, "ollama-info")
        if ollama_info:
            node_info["ollama_info"].append(ollama_info)
        gpu_info = fetch_data_from_node(node_url, "gpu-info")
        if gpu_info:
            node_info["gpu_info"].append(gpu_info)
        aggregated_info.append(node_info)
    return aggregated_info

# Function to get Ollama instance endpoints from a node
def get_ollama_endpoints(node_url):
    ollama_info = fetch_data_from_node(node_url, "ollama-info")
    if ollama_info and "ollama_services" in ollama_info:
        return [service["url"] for service in ollama_info["ollama_services"]]
    return []

# Function to get models from all Ollama instances
def get_models_from_ollama_instances():
    all_models = []
    models_set = set()

    for node in config["nodes"]:
        ollama_endpoints = get_ollama_endpoints(node["url"])
        for endpoint in ollama_endpoints:
            node_models = fetch_data_from_node(endpoint, "api/tags")
            if node_models:
                for model in node_models["models"]:
                    if model["name"] not in models_set:
                        models_set.add(model["name"])
                        all_models.append(model)

    # Sort the models by name in alphabetical order
    all_models_sorted = sorted(all_models, key=lambda x: x["name"])

    return all_models_sorted

# Function to get Ollama instances that have the specified model
def get_ollama_instances_with_model(model_name):
    instances = []

    for node in config["nodes"]:
        ollama_endpoints = get_ollama_endpoints(node["url"])
        for endpoint in ollama_endpoints:
            node_models = fetch_data_from_node(endpoint, "api/tags")
            if node_models:
                for model in node_models["models"]:
                    if model["name"] == model_name:
                        instances.append(endpoint)

    return instances

# Endpoint to get GPU information from all nodes
@app.route('/gpu-info', methods=['GET'])
def gpu_info():
    try:
        gpu_info_list, _ = aggregate_data()
        return jsonify(gpu_info_list)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Endpoint to get Ollama information from all nodes
@app.route('/ollama-info', methods=['GET'])
def ollama_info():
    try:
        _, ollama_info_list = aggregate_data()
        return jsonify(ollama_info_list)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Endpoint to list all nodes with their health status
@app.route('/list-nodes', methods=['GET'])
def list_nodes():
    try:
        nodes_status = [{"node": node["url"], "health": check_node_health(node["url"])} for node in config["nodes"]]
        return jsonify({"nodes": nodes_status})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Endpoint to collect and aggregate hierarchical information
@app.route('/collect-info', methods=['GET'])
def collect_info():
    try:
        aggregated_info = aggregate_hierarchical_data()
        return Response(json.dumps(aggregated_info, indent=4), mimetype='application/json')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# New endpoint to get models from all Ollama instances and de-duplicate
@app.route('/api/tags', methods=['GET'])
def get_models():
    try:
        all_models = get_models_from_ollama_instances()
        return jsonify({"models": all_models})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Endpoint for chat completions with streaming
@app.route('/api/chat', methods=['POST'])
def chat_completion():
    try:
        data = request.json
        model_name = data.get('model')
        
        if model_name not in round_robin_iterators:
            instances = get_ollama_instances_with_model(model_name)
            round_robin_iterators[model_name] = round_robin(instances)
        
        instance_url = next(round_robin_iterators[model_name])
        
        def generate():
            response_stream = fetch_data_from_node(instance_url, 'api/chat', payload=data, method='POST', stream=True)
            if response_stream:
                for chunk in response_stream:
                    yield chunk
            else:
                yield json.dumps({"error": "Failed to get response from Ollama instance"}).encode()

        return Response(stream_with_context(generate()), content_type='application/json')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Endpoint for generate requests with streaming
@app.route('/api/generate', methods=['POST'])
def generate():
    try:
        data = request.json
        model_name = data.get('model')
        
        if model_name not in round_robin_iterators:
            instances = get_ollama_instances_with_model(model_name)
            round_robin_iterators[model_name] = round_robin(instances)
        
        instance_url = next(round_robin_iterators[model_name])
        
        def generate():
            response_stream = fetch_data_from_node(instance_url, 'api/generate', payload=data, method='POST', stream=True)
            if response_stream:
                for chunk in response_stream:
                    yield chunk
            else:
                yield json.dumps({"error": "Failed to get response from Ollama instance"}).encode()

        return Response(stream_with_context(generate()), content_type='application/json')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8009)

