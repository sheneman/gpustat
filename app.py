#
# gpustat - a little API that returns the status of all Nvidia GPUs on a node
#
# Luke Sheneman
# IIDS - University of Idaho
# sheneman@uidaho.edu
# July, 2024
#


from flask import Flask, jsonify, request, Response
import json
import pynvml

app = Flask(__name__)

# Initialize NVML
pynvml.nvmlInit()

def get_gpu_info(index):
    handle = pynvml.nvmlDeviceGetHandleByIndex(index)
    uuid = pynvml.nvmlDeviceGetUUID(handle)
    name = pynvml.nvmlDeviceGetName(handle)
    temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    memory_total = memory_info.total // 1024 // 1024  # bytes to MB
    memory_used = memory_info.used // 1024 // 1024  # bytes to MB

    return {
        "index": index,
        "uuid": uuid,
        "name": name,
        "temperature": temperature,
        "utilization": utilization,
        "memory_total": memory_total,
        "memory_used": memory_used,
        "processes": get_processes(handle)
    }



def get_processes(handle):
    process_info = []
    try:
        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        for process in processes:
            process_info.append({
                "pid": process.pid,
                "process_name": get_process_name(process.pid),
                "used_memory": process.usedGpuMemory // 1024 // 1024  
            })
    except pynvml.NVMLError as e:
        print(f"Failed to get running processes: {e}")

    return process_info



def get_process_name(pid):
    try:
        with open(f"/proc/{pid}/comm", 'r') as file:
            return file.readline().strip()
    except Exception:
        return "Unknown"



@app.route('/gpu-info', methods=['GET'])
def gpu_info():
    try:
        device_count = pynvml.nvmlDeviceGetCount()
        gpu_info_list = [get_gpu_info(i) for i in range(device_count) if get_gpu_info(i) is not None]
        pretty_json = json.dumps(gpu_info_list, indent=4)
        return Response(pretty_json, mimetype='application/json')
    except pynvml.NVMLError as e:
        logger.error(f"Failed to get device count: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)

#@app.teardown_appcontext
#def shutdown_nvml(exception=None):
#    pynvml.nvmlShutdown()

