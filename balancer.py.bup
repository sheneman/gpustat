from flask import Flask, jsonify, request, Response
from flask_cors import CORS
import requests
import json
import time
from datetime import datetime

app = Flask(__name__)
CORS(app)

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

# New endpoint to collect and aggregate hierarchical information
@app.route('/collect-info', methods=['GET'])
def collect_info():
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
    return Response(json.dumps(aggregated_info, indent=4), mimetype='application/json')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8009)

