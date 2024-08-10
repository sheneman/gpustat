from flask import Flask, jsonify, request, Response
from flask_cors import CORS
import requests
import json
import itertools
import logging
from datetime import datetime
import time
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)
CORS(app)

CONFIG_FILE = "nodes.json"
LOG_FILE = "requests.log"

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

# Queue to hold requests when all inference servers are busy
request_queue = Queue()

# Lock for thread-safe queue operations
queue_lock = threading.Lock()

# ThreadPoolExecutor for handling concurrent requests
executor = ThreadPoolExecutor(max_workers=10)  # Adjust the number of workers as needed

# Function to fetch data from a given node
def fetch_data_from_node(node_url, endpoint, payload=None, method='GET', stream=False):
    print(f"Fetching data from {node_url}/{endpoint}")  # Debug print
    try:
        if method == 'POST':
            response = requests.post(f"{node_url}/{endpoint}", json=payload, stream=stream)
        else:
            response = requests.get(f"{node_url}/{endpoint}", stream=stream)
        
        print(f"Response status code: {response.status_code}")  # Debug print
        if response.status_code == 200:
            if stream:
                return response.iter_lines()
            return response.json()
        else:
            print(f"Error response: {response.text}")  # Debug print
            return None
    except Exception as e:
        print(f"Failed to fetch data from {node_url}: {e}")
        return None

# Function to check the health of a node
def check_node_health(node_url):
    print(f"Checking health of {node_url}")  # Debug print
    try:
        response = requests.get(f"{node_url}/health")
        return response.status_code == 200
    except Exception as e:
        print(f"Failed to check health of {node_url}: {e}")
        return False

# Function to get Ollama instance endpoints from a node
def get_ollama_endpoints(node_url):
    print(f"Getting Ollama endpoints for {node_url}")  # Debug print
    ollama_info = fetch_data_from_node(node_url, "ollama-info")
    if ollama_info and "ollama_services" in ollama_info:
        endpoints = [service["url"] for service in ollama_info["ollama_services"]]
        print(f"Found endpoints: {endpoints}")  # Debug print
        return endpoints
    print(f"No Ollama services found for {node_url}")  # Debug print
    return []

# Function to get Ollama instances that have the specified model
def get_ollama_instances_with_model(model_name):
    print(f"Getting Ollama instances for model: {model_name}")  # Debug print
    instances = []

    for node in config["nodes"]:
        print(f"Checking node: {node['url']}")  # Debug print
        ollama_endpoints = get_ollama_endpoints(node["url"])
        for endpoint in ollama_endpoints:
            print(f"Checking endpoint: {endpoint}")  # Debug print
            node_models = fetch_data_from_node(endpoint, "api/tags")
            if node_models:
                print(f"Models found: {node_models}")  # Debug print
                for model in node_models.get("models", []):
                    if model["name"] == model_name:
                        instances.append(endpoint)
                        print(f"Added instance: {endpoint}")  # Debug print

    print(f"Instances with model {model_name}: {instances}")  # Debug print
    return instances

# Function to log requests in jsonl format
def log_request(log_data):
    with open(LOG_FILE, 'a') as log_file:
        log_file.write(json.dumps(log_data) + '\n')

# Function to extract prompt length from the request data
def extract_prompt_length(messages):
    prompt = ' '.join([msg.get('content', '') for msg in messages])
    return len(prompt.split())

# Function to process a single request
def process_single_request(request_data):
    print("process_single_request()")  # Keep this print statement
    print(request_data)

    data = request_data['data']
    model_name = request_data['model_name']
    client_ip = request_data['client_ip']
    request_time = request_data['request_time']

    print(f"Processing request for model: {model_name}")  # Debug print

    if model_name not in round_robin_iterators:
        print(f"Creating new round-robin iterator for {model_name}")  # Debug print
        instances = get_ollama_instances_with_model(model_name)
        if not instances:
            print(f"No instances found for model {model_name}")  # Debug print
            yield json.dumps({"error": f"No instances available for model {model_name}"}) + '\n'
            return
        round_robin_iterators[model_name] = round_robin(instances)

    instance_url = next(round_robin_iterators[model_name])
    model_loaded = instance_url is not None

    print(f"Selected instance: {instance_url}")  # Debug print

    start_time = time.time()

    response_content = []
    response_stream = fetch_data_from_node(instance_url, 'api/chat', payload=data, method='POST', stream=True)
    if response_stream:
        for line in response_stream:
            if line:
                try:
                    parsed_line = json.loads(line)
                    response_content.append(parsed_line)
                    yield json.dumps(parsed_line) + '\n'
                except json.JSONDecodeError:
                    print(f"Error parsing JSON: {line}")
        completion_tokens = sum(len(resp.get('content', '').split()) for resp in response_content if 'content' in resp)
    else:
        completion_tokens = 0
        error_response = json.dumps({"error": "Failed to get response from Ollama instance"})
        yield error_response + '\n'

    end_time = time.time()
    total_tokens = extract_prompt_length(data.get('messages', [])) + completion_tokens
    log_data = {
        "timestamp": request_time,
        "ip_address": client_ip,
        "model": model_name,
        "prompt_tokens": extract_prompt_length(data.get('messages', [])),
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "inference_server": instance_url,
        "model_loaded": model_loaded,
        "response_time_ms": int((end_time - start_time) * 1000)
    }
    log_request(log_data)

# Function to process the request queue
def process_queue():
    print("process_queue()")  # Keep this print statement
    while True:
        request_data = None
        with queue_lock:
            print("HERE A - request_queue has length: %d" % request_queue.qsize())  # keep this print statement
            if not request_queue.empty():
                request_data = request_queue.get()
        
        if request_data:
            executor.submit(process_single_request, request_data)
        
        time.sleep(1.1)  # Reduced sleep time for more frequent checks

# Function to run process_queue in a separate thread
def run_queue_processor():
    threading.Thread(target=process_queue, daemon=True).start()

# Start the queue processing
run_queue_processor()

# Function to aggregate data from all nodes
def aggregate_data():
    print("Aggregating data from all nodes")  # Debug print
    gpu_info_list = []
    ollama_info_list = []

    for node in config["nodes"]:
        print(f"Fetching data from node: {node['url']}")  # Debug print
        node_gpu_info = fetch_data_from_node(node["url"], "gpu-info")
        node_ollama_info = fetch_data_from_node(node["url"], "ollama-info")
        
        if node_gpu_info:
            gpu_info_list.append({"node": node["url"], "gpu_info": node_gpu_info})
        if node_ollama_info:
            ollama_info_list.append({"node": node["url"], "ollama_info": node_ollama_info})

    print(f"Aggregated data: {len(gpu_info_list)} GPU info, {len(ollama_info_list)} Ollama info")  # Debug print
    return gpu_info_list, ollama_info_list

# Function to collect and aggregate hierarchical information
def aggregate_hierarchical_data():
    print("Aggregating hierarchical data")  # Debug print
    aggregated_info = []
    for node in config["nodes"]:
        node_url = node["url"]
        print(f"Collecting data for node: {node_url}")  # Debug print
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
    print(f"Aggregated info for {len(aggregated_info)} nodes")  # Debug print
    return aggregated_info

# Function to get models from all Ollama instances
def get_models_from_ollama_instances():
    print("Getting models from all Ollama instances")  # Debug print
    all_models = []
    models_set = set()

    for node in config["nodes"]:
        ollama_endpoints = get_ollama_endpoints(node["url"])
        for endpoint in ollama_endpoints:
            print(f"Fetching models from endpoint: {endpoint}")  # Debug print
            node_models = fetch_data_from_node(endpoint, "api/tags")
            if node_models:
                for model in node_models.get("models", []):
                    if model["name"] not in models_set:
                        models_set.add(model["name"])
                        all_models.append(model)

    # Sort the models by name in alphabetical order
    all_models_sorted = sorted(all_models, key=lambda x: x["name"])
    print(f"Total unique models found: {len(all_models_sorted)}")  # Debug print

    return all_models_sorted

# Endpoint for chat completions with streaming
@app.route('/api/chat', methods=['POST'])
def chat_completion():
    try:
        print("Received chat completion request")  # Debug print
        data = request.json
        model_name = data.get('model')
        client_ip = request.remote_addr
        request_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(f"Request data: {data}")  # Debug print

        def generate():
            for chunk in process_single_request({
                "data": data,
                "model_name": model_name,
                "client_ip": client_ip,
                "request_time": request_time
            }):
                yield chunk

        return Response(generate(), mimetype='application/json', content_type='application/json')
    except Exception as e:
        print(f"Error in chat_completion: {str(e)}")  # Debug print
        return jsonify({"error": str(e)}), 500

# Endpoint for generate requests with streaming
@app.route('/api/generate', methods=['POST'])
def generate():
    try:
        print("Received generate request")  # Debug print
        data = request.json
        model_name = data.get('model')
        client_ip = request.remote_addr
        request_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(f"Request data: {data}")  # Debug print

        def generate():
            for chunk in process_single_request({
                "data": data,
                "model_name": model_name,
                "client_ip": client_ip,
                "request_time": request_time
            }):
                yield chunk

        return Response(generate(), mimetype='application/json', content_type='application/json')
    except Exception as e:
        print(f"Error in generate: {str(e)}")  # Debug print
        return jsonify({"error": str(e)}), 500

# Endpoint for OpenAI-style chat completions with streaming
@app.route('/v1/chat/completions', methods=['POST'])
def openai_chat_completions():
    try:
        print("Received OpenAI-style chat completion request")  # Debug print
        data = request.json
        model_name = data.get('model')
        client_ip = request.remote_addr
        request_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(f"Request data: {data}")  # Debug print

        def generate():
            for chunk in process_single_request({
                "data": data,
                "model_name": model_name,
                "client_ip": client_ip,
                "request_time": request_time
            }):
                yield chunk

        return Response(generate(), mimetype='application/json', content_type='application/json')
    except Exception as e:
        print(f"Error in openai_chat_completions: {str(e)}")  # Debug print
        return jsonify({"error": str(e)}), 500

# Endpoint to get GPU information from all nodes
@app.route('/gpu-info', methods=['GET'])
def gpu_info():
    try:
        print("Received request for GPU info")  # Debug print
        gpu_info_list, _ = aggregate_data()
        print(f"Returning GPU info for {len(gpu_info_list)} nodes")  # Debug print
        pretty_json = json.dumps(gpu_info_list, indent=4)
        return Response(pretty_json, mimetype='application/json')
    except Exception as e:
        print(f"Error in gpu_info: {str(e)}")  # Debug print
        error_json = json.dumps({"error": str(e)}, indent=4)
        return Response(error_json, mimetype='application/json'), 500

# Endpoint to get Ollama information from all nodes
@app.route('/ollama-info', methods=['GET'])
def ollama_info():
    try:
        print("Received request for Ollama info")  # Debug print
        _, ollama_info_list = aggregate_data()
        print(f"Returning Ollama info for {len(ollama_info_list)} nodes")  # Debug print
        pretty_json = json.dumps(ollama_info_list, indent=4)
        return Response(pretty_json, mimetype='application/json')
    except Exception as e:
        print(f"Error in ollama_info: {str(e)}")  # Debug print
        error_json = json.dumps({"error": str(e)}, indent=4)
        return Response(error_json, mimetype='application/json'), 500

# Endpoint to list all nodes with their health status
@app.route('/list-nodes', methods=['GET'])
def list_nodes():
    try:
        print("Received request to list nodes")  # Debug print
        nodes_status = [{"node": node["url"], "health": check_node_health(node["url"])} for node in config["nodes"]]
        print(f"Returning status for {len(nodes_status)} nodes")  # Debug print
        pretty_json = json.dumps({"nodes": nodes_status}, indent=4)
        return Response(pretty_json, mimetype='application/json')
    except Exception as e:
        print(f"Error in list_nodes: {str(e)}")  # Debug print
        error_json = json.dumps({"error": str(e)}, indent=4)
        return Response(error_json, mimetype='application/json'), 500

# Endpoint to collect and aggregate hierarchical information
@app.route('/collect-info', methods=['GET'])
def collect_info():
    try:
        print("Received request to collect hierarchical info")  # Debug print
        aggregated_info = aggregate_hierarchical_data()
        print(f"Returning aggregated info for {len(aggregated_info)} nodes")  # Debug print
        pretty_json = json.dumps(aggregated_info, indent=4)
        return Response(pretty_json, mimetype='application/json')
    except Exception as e:
        print(f"Error in collect_info: {str(e)}")  # Debug print
        error_json = json.dumps({"error": str(e)}, indent=4)
        return Response(error_json, mimetype='application/json'), 500

# Endpoint to get models from all Ollama instances and de-duplicate
@app.route('/api/tags', methods=['GET'])
def get_models():
    try:
        print("Received request for model tags")  # Debug print
        all_models = get_models_from_ollama_instances()
        print(f"Returning {len(all_models)} unique models")  # Debug print
        pretty_json = json.dumps({"models": all_models}, indent=4)
        return Response(pretty_json, mimetype='application/json')
    except Exception as e:
        print(f"Error in get_models: {str(e)}")  # Debug print
        error_json = json.dumps({"error": str(e)}, indent=4)
        return Response(error_json, mimetype='application/json'), 500

if __name__ == '__main__':
    print("Starting the Flask application")  # Debug print
    app.run(host='0.0.0.0', port=8009, threaded=True)
    print("Flask application has stopped")  # Debug print

