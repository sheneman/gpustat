from flask import Flask, jsonify, request, Response
from flask_cors import CORS
import requests
import json
import itertools
import random
import logging
from datetime import datetime
import threading
import time

app = Flask(__name__)
CORS(app)

CONFIG_FILE = "nodes.json"
LOG_FILE = "requests.log"

global_cluster_state = {}
global_thread_started = False

REFRESH_INTERVAL = 60     # seconds


# Load configuration
with open(CONFIG_FILE, 'r') as config_file:
    config = json.load(config_file)


def parse_memory(memory_str):
    """Convert memory string to MB."""
    if "GB" in memory_str:
        return int(float(memory_str.replace(" GB", "")) * 1024)
    elif "MB" in memory_str:
        return int(memory_str.replace(" MB", ""))
    return 0


# Function to fetch data from a given node
def fetch_data_from_node(node_url, endpoint, payload=None, method='GET', stream=False):
    print(f"Fetching data from {node_url}/{endpoint}")  # Debug print
    try:
        if method == 'POST':
            response = requests.post(f"{node_url}/{endpoint}", json=payload, stream=stream)
        else:
            response = requests.get(f"{node_url}/{endpoint}", stream=stream)
        
        if response.status_code == 200:
            if stream:
                return response.iter_lines()
            return response.json()
        else:
            print(f"Response status code: {response.status_code}")  # Debug print
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

# Function to get Ollama instance endpoints from a node and associate with agent endpoint
def get_ollama_endpoints(node_url):
    print(f"Getting Ollama endpoints for {node_url}")  # Debug print
    ollama_info = fetch_data_from_node(node_url, "ollama-info")
    if ollama_info and "ollama_services" in ollama_info:
        endpoints = {service["url"]: node_url for service in ollama_info["ollama_services"]}
        print(f"Found endpoints: {endpoints}")  # Debug print
        return endpoints
    print(f"No Ollama services found for {node_url}")  # Debug print
    return {}



def get_optimal_ollama_instance_with_model(model_name):
    global global_cluster_state

    print("###########################")
    print(f"Getting Ollama instances for model: {model_name}")  # Debug print
    utilization_threshold = 25.0

    viable_endpoints = []

    for node in global_cluster_state:
        for ollama_info in node.get("ollama_info", []):
            for service in ollama_info.get("ollama_services", []):
                # Check if the model is available
                for model in service.get("models_avail", []):
                    if model_name == model['name']:
                        # Calculate total GPU memory available for the service
                        total_memory = 0
                        used_memory = 0
                        gpu_utilizations = []
   
                        for gpu in node.get("gpu_info", [])[0].get("gpus", []):
                            if gpu['index'] in service.get("gpu_indices", []):
                                total_memory += parse_memory(gpu['memory_total'])
                                used_memory += parse_memory(gpu['memory_used'])
                                gpu_utilizations.append(gpu['utilization'])
   
                        available_memory = total_memory - used_memory
   
                        # Get the memory requirement of the model
                        model_memory_required = parse_memory(model['size'])
   
                        print("MEM AVAIL: ", available_memory, " MEM REQUIRED: ", model_memory_required)
   
                        # Check if the model is currently running
                        running_models = [running_model['name'] for running_model in service.get("models_running", [])]
                        is_running = model_name in running_models
   
                        # Calculate average GPU utilization
                        average_gpu_utilization = sum(gpu_utilizations) / len(gpu_utilizations) if gpu_utilizations else 0
   
                        # Create the endpoint_info dictionary
                        endpoint_info = {
                            "service_name": service["service_name"],
                            "url": service["url"],
                            "total_gpu_memory_MB": total_memory,
                            "available_gpu_memory_MB": available_memory,
                            "model_memory_required_MB": model_memory_required,
                            "average_gpu_utilization": average_gpu_utilization,
                            "running": is_running
                        } 
                        viable_endpoints.append(endpoint_info)
    
    # Find the optimal endpoint
    if viable_endpoints:
        # Filter viable endpoints by available memory
        memory_sufficient_endpoints = [
            endpoint for endpoint in viable_endpoints
            if endpoint["available_gpu_memory_MB"] >= endpoint["model_memory_required_MB"]
        ]

        if memory_sufficient_endpoints:
            # Sort by available memory in descending order
            memory_sufficient_endpoints.sort(key=lambda x: x["available_gpu_memory_MB"], reverse=True)

            # Select the best endpoint with the lowest GPU utilization under the threshold
            for endpoint in memory_sufficient_endpoints:
                if endpoint["average_gpu_utilization"] <= utilization_threshold:
                    print(f"Selected endpoint: {endpoint}")
                    return endpoint["url"]

            # If all have high utilization, select the one with the most available memory
            best_endpoint = memory_sufficient_endpoints[0]
            print(f"Selected endpoint (high utilization): {best_endpoint}")
            return best_endpoint["url"]
        else:
            # If no endpoints have sufficient memory, choose one at random
            random_endpoint = random.choice(viable_endpoints)
            print(f"Selected endpoint (insufficient memory): {random_endpoint}")
            return random_endpoint["url"]
    else:
        print("No viable endpoints found.")
        return None




# Function to get running models on an Ollama instance
def get_running_models(instance_url):
    running_models = fetch_data_from_node(instance_url, "api/ps")
    if running_models and "models" in running_models:
        return [model["name"] for model in running_models["models"]]
    return []

# Function to get GPU utilization of an instance
def get_gpu_utilization(agent_url):
    gpu_info = fetch_data_from_node(agent_url, "gpu-info")
    if gpu_info:
        try:
            print(f"GPU info received: {gpu_info}")  # Debug print
            
            gpus = gpu_info.get("gpus", [])
            if not gpus:
                print("No GPU information available.")
                return float('inf')

            # Calculate average utilization across all GPUs
            total_utilization = sum(gpu.get("utilization", 0) for gpu in gpus)
            average_utilization = total_utilization / len(gpus)

            return average_utilization
        except KeyError as e:
            print(f"Error parsing GPU utilization: {e}")
            return float('inf')
    else:
        print(f"Failed to retrieve GPU utilization from {agent_url}.")
        return float('inf')  # Return a high value if GPU utilization cannot be fetched


# Function to log requests in jsonl format
def log_request(log_data):
    with open(LOG_FILE, 'a') as log_file:
        log_file.write(json.dumps(log_data) + '\n')

# Function to extract prompt length from the request data
def extract_prompt_length(messages):
    prompt = ' '.join([msg.get('content', '') for msg in messages])
    return len(prompt.split())


def process_single_request(request_data):
    print("process_single_request()")

    data = request_data['data']
    model_name = request_data['model_name']
    client_ip = request_data['client_ip']
    request_time = request_data['request_time']
    request_path = request_data['request_path']

    print(f"Processing request for model: {model_name}")

    # Get the optimal instance with the new greedy approach
    instance_url = get_optimal_ollama_instance_with_model(model_name)
    print("LUKE:", instance_url)
    if not instance_url:
        print(f"No instances available for model {model_name}")
        yield '{"error": "No instances available for model"}\n'
        return

    print(f"Selected instance: {instance_url}")

    start_time = time.time()

    # Determine the backend endpoint based on the request path
    if '/v1/chat/completions' in request_path:
        backend_endpoint = 'v1/chat/completions'
        format_sse = True  # Flag to determine if Server-Sent Events (SSE) formatting is needed
        is_v1_endpoint = True  # Flag to identify /v1/chat/completions
    elif '/api/embeddings' in request_path:
        backend_endpoint = 'api/embeddings'
        format_sse = False  # No SSE formatting needed
        is_v1_endpoint = False
    else:
        backend_endpoint = 'api/chat'
        format_sse = False  # No SSE formatting needed
        is_v1_endpoint = False

    response_content = []
    response_stream = fetch_data_from_node(instance_url, backend_endpoint, payload=data, method='POST', stream=True)

    if response_stream:
        try:
            for line in response_stream:
                if line:
                    line = line.decode('utf-8').strip()

                    if format_sse:
                        # For SSE format (used by /v1/chat/completions)
                        if line.startswith("data:"):
                            line = line[len("data:"):].strip()  # Remove "data:" prefix

                        if line == "[DONE]":
                            print("Received end-of-stream marker")
                            yield 'data: [DONE]\n\n'
                            break

                        try:
                            parsed_line = json.loads(line)
                            response_content.append(parsed_line)
                            yield f"data: {json.dumps(parsed_line)}\n\n"
                        except json.JSONDecodeError:
                            print(f"Non-JSON or empty line received: {line}")
                            yield f"data: {line}\n\n"
                    else:
                        # For /api/chat and /api/embed (no SSE format)
                        try:
                            parsed_line = json.loads(line)
                            response_content.append(parsed_line)
                            # Yield the JSON object with a newline separator
                            yield f"{json.dumps(parsed_line)}\n"
                        except json.JSONDecodeError:
                            print(f"Non-JSON or empty line received: {line}")
                            yield f"{line}\n"

        except Exception as e:
            print(f"Error during streaming: {str(e)}")
            if format_sse:
                yield f"data: [ERROR] Streaming interrupted: {str(e)}\n\n"
                yield 'data: [DONE]\n\n'
            else:
                yield f'{{"error": "Streaming interrupted: {str(e)}"}}\n'
    else:
        print("No response stream available from node")
        if format_sse:
            yield 'data: {"error": "Failed to get response from LLM instance"}\n\n'
        else:
            yield '{"error": "Failed to get response from LLM instance"}\n'

    end_time = time.time()

    # Initialize token counts
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0

    # Extract token counts from the final response
    if response_content:
        final_response = response_content[-1]

        if is_v1_endpoint:
            # For /v1/chat/completions endpoint
            if "usage" in final_response:
                prompt_tokens = final_response["usage"].get("prompt_tokens", 0)
                completion_tokens = final_response["usage"].get("completion_tokens", 0)
                total_tokens = final_response["usage"].get("total_tokens", prompt_tokens + completion_tokens)
        else:
            # For /api/chat endpoint
            prompt_tokens = final_response.get("prompt_eval_count", 0)
            completion_tokens = final_response.get("eval_count", 0)
            total_tokens = prompt_tokens + completion_tokens

    log_data = {
        "timestamp": request_time,
        "ip_address": client_ip,
        "model": model_name,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "inference_server": instance_url,
        "model_loaded": True,
        "response_time_ms": int((end_time - start_time) * 1000)
    }
    log_request(log_data)




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
    print(f"Aggregated info for {len(aggregated_info)} nodes\n")  # Debug print
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


@app.route('/api/chat', methods=['POST'])
def chat_completion():
    try:
        print("Received chat completion request")
        data = request.json
        model_name = data.get('model')
        client_ip = request.remote_addr
        request_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        request_path = request.path

        def generate():
            try:
                for chunk in process_single_request({
                    "data": data,
                    "model_name": model_name,
                    "client_ip": client_ip,
                    "request_time": request_time,
                    "request_path": request_path
                }):
                    yield chunk
            except GeneratorExit:
                print("Client closed connection prematurely")
            except Exception as e:
                print(f"Unexpected error in streaming: {str(e)}")
                yield f'{{"error": "Unexpected error: {str(e)}"}}\n'

        return Response(generate(), mimetype='application/json', content_type='application/json')
    except Exception as e:
        print(f"Error in chat_completion: {str(e)}")
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

        #print(f"Request data: {data}")  # Debug print

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


# In the openai_chat_completions function
@app.route('/v1/chat/completions', methods=['POST'])
def openai_chat_completions():
    try:
        print("Received OpenAI-style chat completion request")  # Debug print
        data = request.json
        model_name = data.get('model')
        client_ip = request.remote_addr
        request_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        request_path = request.path  # Capture the request path

        #print(f"Request data: {data}")  # Debug print

        def generate():
            for chunk in process_single_request({
                "data": data,
                "model_name": model_name,
                "client_ip": client_ip,
                "request_time": request_time,
                "request_path": request_path  # Include the request path in the request data
            }):
                yield chunk

        return Response(generate(), mimetype='application/json', content_type='application/json')
    except Exception as e:
        print(f"Error in openai_chat_completions: {str(e)}")  # Debug print
        return jsonify({"error": str(e)}), 500

@app.route('/api/embeddings', methods=['POST'])
def embed_text():
    try:
        print("Received embedding request")
        data = request.json
        model_name = data.get('model')
        client_ip = request.remote_addr
        request_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        request_path = request.path

        def generate():
            try:
                for chunk in process_single_request({
                    "data": data,
                    "model_name": model_name,
                    "client_ip": client_ip,
                    "request_time": request_time,
                    "request_path": request_path
                }):
                    yield chunk
            except GeneratorExit:
                print("Client closed connection prematurely")
            except Exception as e:
                print(f"Unexpected error in streaming: {str(e)}")
                yield f'{{"error": "Unexpected error: {str(e)}"}}\n'

        return Response(generate(), mimetype='application/json', content_type='application/json')
    except Exception as e:
        print(f"Error in embed_text(): {str(e)}")
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


def start_background_thread():
    print("start_background_thread()")
    global global_thread_started
    if not global_thread_started:
        global_thread_started = True
        threading.Thread(target=update_ollama_state_periodically, args=(REFRESH_INTERVAL,), daemon=True).start()


def update_ollama_state_periodically(delay=REFRESH_INTERVAL):
    global global_cluster_state
    while True:
        global_cluster_state = aggregate_hierarchical_data()
        time.sleep(delay)  # Wait for the specified delay before updating again


start_background_thread()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8009, threaded=True)

