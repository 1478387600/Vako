import requests
import json
import time
import random
from tqdm import tqdm

# ----------------- 配置部分 -----------------
API_KEY = "sk-67d92d294cb644ca8841af0389369558"  # <-- 替换成你的API KEY
API_URL = "https://api.deepseek.com/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# 设备与传感器清单
DEVICES = [
    "living_room_ac", "bedroom_ac", "main_purifier",
    "living_room_curtain", "kitchen_blind",
    "kitchen_light", "bedroom_light", "hallway_light",
    "living_room_tv", "bathroom_window", "bedroom_window"
]
SENSORS = [
    "indoor_temp_sensor", "outdoor_temp_sensor",
    "power_meter", "rain_sensor"
]

# System Prompt（始终存在的角色设定）
SYSTEM_PROMPT = (
    "You are an AI assistant for a smart home system. "
    "When receiving a user instruction, you must respond in strict JSON format. "
    "The JSON must contain 4 fields: 'message', 'type', 'name', 'arguments'.\n\n"
    "If the user gives a control instruction (e.g., turn on, open curtain), respond like:\n"
    "{\n"
    '  "message": "OK, I will turn on the TV.",\n'
    '  "type": "tool",\n'
    '  "name": "control_device",\n'
    '  "arguments": {"device_id": "living_room_tv", "status": "on", "level": null}\n'
    "}\n\n"
    "If the user asks for a sensor status (e.g., check temperature), respond like:\n"
    "{\n"
    '  "message": "Checking the indoor temperature sensor.",\n'
    '  "type": "tool",\n'
    '  "name": "get_status",\n'
    '  "arguments": {"device_id": "indoor_temp_sensor"}\n'
    "}\n\n"
    "Only return JSON without any other text."
)

# ----------------- 核心函数部分 -----------------

# 生成用户指令 prompt
def generate_user_instruction():
    device_list = ", ".join(DEVICES)
    sensor_list = ", ".join(SENSORS)

    meta_prompt_text = (
        f"You are creating a smart home instruction. "
        f"The available devices are: {device_list}. "
        f"The available sensors are: {sensor_list}. "
        f"Generate a realistic user instruction involving either device control (turn on, off, open, close) "
        f"or sensor query (get temperature, humidity, rain). "
        f"Return ONLY the instruction text without explanation."
        f"You are generating a smart home user instruction. " 
        f"The available devices are: {DEVICES}. " 
        f"The available sensors are: {SENSORS}. " 
        f"Generate a realistic instruction involving device control (e.g., turn on, set temperature, open curtain) " 
        f"The instruction should be simple in a single sentence. Do not use conditional statements." 
        f"When the user mention a device or sensor, he uses the natural language instead of id of that one." 
        f"Instructions are mainly about control the devices, turn on/off the devices, query the state, etc." 
        f"Only the light, air conditioner, blind have the value to set, others just on or off." 
        f"or sensor query (e.g., check temperature, get rain status). " 
        f"Return only the instruction text, without explanations."
    )

    meta_prompt = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": meta_prompt_text}],
        "temperature": 0.8,
        "top_p": 0.95,
        "max_tokens": 100
    }
    response = requests.post(API_URL, headers=HEADERS, data=json.dumps(meta_prompt))
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"].strip('"').strip()
    else:
        print(f"[Meta Prompt Error] {response.status_code}: {response.text}")
        return None

# 生成 Assistant 响应（携带 System Prompt + Few-shot示例）
def generate_assistant_response(user_prompt):
    few_shot_examples = [
        {"role": "user", "content": "Please turn on the TV."},
        {"role": "assistant", "content": json.dumps({
            "message": "OK, I will turn on the TV.",
            "type": "tool",
            "name": "control_device",
            "arguments": {
                "device_id": "living_room_tv",
                "status": "on",
                "level": None
            }
        })},
        {"role": "user", "content": "What's the indoor temperature?"},
        {"role": "assistant", "content": json.dumps({
            "message": "Checking the indoor temperature sensor.",
            "type": "tool",
            "name": "get_status",
            "arguments": {
                "device_id": "indoor_temp_sensor"
            }
        })}
    ]

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + few_shot_examples + [{"role": "user", "content": user_prompt}]

    data = {
        "model": "deepseek-chat",
        "messages": messages,
        "temperature": 0.5,
        "top_p": 0.9,
        "max_tokens": 512
    }
    response = requests.post(API_URL, headers=HEADERS, data=json.dumps(data))
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        print(f"[Assistant Response Error] {response.status_code}: {response.text}")
        return None

# 模拟 Tool 返回反馈
def simulate_tool_feedback(tool_call):
    if tool_call["name"] == "get_status":
        device_id = tool_call["arguments"].get("device_id")
        if device_id in SENSORS:
            # 假设返回传感器数值
            value = round(20 + 10 * random.random(), 1)
            return {
                "id": device_id,
                "type": "sensor",
                "status": str(value),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
            }
        else:
            return f"Device {device_id} status unknown."
    else:
        device_id = tool_call["arguments"].get("device_id")
        status = tool_call["arguments"].get("status")
        return f"Device {device_id} turned {status}"

# 创建 instruct 和 feedback 两种条目
def create_instruction_feedback(user_prompt, assistant_response):
    try:
        assistant_content = json.loads(assistant_response)
    except json.JSONDecodeError:
        raise ValueError("Assistant返回格式错误，不是合法JSON！")

    instruction_entry = {
        "type": "instruct",
        "conversations": [
            {"from": "user", "content": user_prompt},
            {"from": "assistant", "content": assistant_content}
        ]
    }

    feedback_content = simulate_tool_feedback(assistant_content)

    feedback_entry = {
        "type": "feedback",
        "conversations": [
            {"from": "tool", "content": feedback_content},
            {"from": "assistant", "content": {
                "message": f"Done. {assistant_content['message']}" if isinstance(feedback_content, str)
                           else f"The {feedback_content['type']} {feedback_content['id']} reports {feedback_content['status']}."
            }}
        ]
    }

    return [instruction_entry, feedback_entry]

# 主程序
if __name__ == "__main__":
    dataset = []
    num_samples = 10

    print(f"🚀 开始生成指令与反馈，共需生成 {num_samples} 条...\n")

    for i in range(1, num_samples + 1):
        print(f"📌 [{i}/{num_samples}] 正在生成用户指令...")
        user_prompt = generate_user_instruction()
        if not user_prompt:
            print("❌ 用户指令生成失败，跳过该样本。\n")
            continue

        print(f"➡️ 用户指令: {user_prompt}")

        print("💬 正在获取 Assistant 响应...")
        assistant_response = generate_assistant_response(user_prompt)
        print(f"Assistant 响应{assistant_response}")
        if not assistant_response:
            print("❌ Assistant 响应失败，跳过该样本。\n")
            continue

        try:
            print("🔧 构建对话与反馈条目...")
            entries = create_instruction_feedback(user_prompt, assistant_response)
            dataset.extend(entries)
            print("✅ 成功添加到数据集中。\n")
        except Exception as e:
            print(f"[⚠️ Error] {e}\n")

        time.sleep(0.8)

    with open("generated_instructions.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print("🎉 全部生成完毕，文件已保存为 generated_instructions.json")
