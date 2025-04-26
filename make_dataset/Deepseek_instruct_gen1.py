import requests
import json
import time
import random
from tqdm import tqdm
import uuid

API_KEY = "sk-67d92d294cb644ca8841af0389369558"
API_URL = "https://api.deepseek.com/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# 可用设备与传感器
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

# System prompt + few-shot 示例（含 instruct 和 feedback）
SYSTEM_PROMPT = (
    "You are an AI assistant for a smart home system. "
    "Your job is to interpret user instructions and respond using structured JSON. "
    "You will also summarize the tool's feedback in natural language.\n\n"
    "Enhance the diversity of instructions and answers, including sentence patterns, grammatical structures, and word use."
    "Assistant must return a JSON object for instructions, like:\n"
    "{\n"
    '  "message": "OK, I will turn on the TV.",\n'
    '  "type": "tool",\n'
    '  "name": "control_device",\n'
    '  "arguments": {"device_id": "living_room_tv", "status": "on", "level": null}\n'
    "}\n\n"
    "For tool feedback, you will receive a JSON object (e.g., sensor reading or device result) "
    "and generate a natural sentence based on it. Only respond with that sentence."
    "Do not actually invoke any tool calling or use OpenAI tool calling format."
)

FEW_SHOTS = [
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
    }, ensure_ascii=False)},

    {"role": "user", "content": "What's the temperature indoors?"},
    {"role": "assistant", "content": json.dumps({
        "message": "Checking the indoor temperature sensor.",
        "type": "tool",
        "name": "get_status",
        "arguments": {
            "device_id": "indoor_temp_sensor"
        }
    }, ensure_ascii=False)}
]


# 清洗 assistant 响应，去除 ```json``` 代码块等包装
def clean_json_response(response_str):
    response_str = response_str.strip()
    if response_str.startswith("```json"):
        response_str = response_str.lstrip("```json").rstrip("```").strip()
    elif response_str.startswith("```"):
        response_str = response_str.lstrip("```").rstrip("```").strip()
    return response_str

generated_prompts = []

# 随机生成用户指令
def generate_user_instruction():

    already_generated = "\n".join(f"- {p}" for p in generated_prompts) if generated_prompts else "None yet."

    meta_prompt_text = (
        f"The available devices are: {', '.join(DEVICES)}. "
        f"The available sensors are: {', '.join(SENSORS)}. "
        f"Generate a natural smart home instruction that involves either controlling a device or querying a sensor."
        f"Generate a realistic user instruction involving either device control (turn on, off, open, close) "
        f"or sensor query (get temperature, humidity, rain). "
        f"Return ONLY the instruction text without explanation."
        f"You are generating a smart home user instruction. " 
        f"Generate a realistic instruction involving device control (e.g., turn on, set temperature, open curtain) "
        f"Generate simple instructions that has no clause, and once related to only one deivce or sensor."
        f"The instructions should be polite, in a daily everyday natural language."
        f"Try to enhance the diversity of instructions, including sentence patterns, grammatical structures, and word use."
        f"The instruction should be simple in a single sentence. Do not use conditional statements." 
        f"When the user mention a device or sensor, he uses the natural language instead of id of that one." 
        f"Instructions are mainly about control the devices, turn on/off the devices, query the state, etc." 
        f"Only the light, air conditioner, blind have the value to set, others just on or off." 
        f"All the level are int numbers."
        f"TV has only status that is either on or off, no other attributes such as channel."
        f"Air conditioners have only on or off, and the level that is temperature. They have no other attributes like mode."
        f"Lights has no color, only on or off and level."
        f"Do not make an instruction related to a device that is not in the available devices or sensors"
        f"Use air conditioner instead of AC."
        f"or sensor query (e.g., check temperature, get rain status). " 
        f"Return only the instruction text, without explanations."  
        f"Instructions already generated:\n{already_generated}\n\n"
        f"Please create a new and realistic instruction that is not a duplicate or paraphrase of the above."
    )
    data = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": meta_prompt_text}],
        "temperature": 0.9,
        "top_p": 0.95,
        "max_tokens": 100
    }
    response = requests.post(API_URL, headers=HEADERS, data=json.dumps(data))
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"].strip()
    else:
        print(f"[Meta Prompt Error] {response.status_code}: {response.text}")
        return None

# assistant 生成工具调用 JSON
def generate_assistant_response(user_prompt):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + FEW_SHOTS + [{"role": "user", "content": user_prompt}]
    data = {
        "model": "deepseek-chat",
        "messages": messages,
        "temperature": 0.5,
        "top_p": 0.9,
        "max_tokens": 512
    }
    response = requests.post(API_URL, headers=HEADERS, data=json.dumps(data))
    if response.status_code == 200:
        raw = response.json()["choices"][0]["message"]["content"]
        return clean_json_response(raw)
    else:
        print(f"[Assistant Response Error] {response.status_code}: {response.text}")
        return None

# 模拟工具反馈（类型化处理）
def simulate_tool_feedback(tool_call):
    device_id = tool_call["arguments"].get("device_id")
    ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
    if tool_call["name"] == "get_status":
        value = round(random.uniform(18, 30), 1)
        return {
            "id": device_id,
            "type": "sensor",
            "status": f"{value}",
            "timestamp": ts
        }
    else:
        status = tool_call["arguments"].get("status")
        return {
            "id": device_id,
            "type": "device",
            "status": f"Device {device_id} turned {status}",
            "timestamp": ts
        }

# 调用 assistant 让其根据 tool 反馈生成自然语言
def generate_feedback_natural_sentence(tool_feedback):
    prompt = (
        f"The following is the status information returned from the smart home device or sensor:\n\n"
        f"{json.dumps(tool_feedback, ensure_ascii=False, indent=2)}\n\n"
        f"Please summarize it into a natural English sentence for the user."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *FEW_SHOTS,
        {"role": "user", "content": prompt}  # ⬅️ 重点：是 user，不是 tool
    ]
    data = {
        "model": "deepseek-chat",
        "messages": messages,
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 150
    }
    response = requests.post(API_URL, headers=HEADERS, data=json.dumps(data))
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"].strip()
    else:
        print(f"[Feedback Natural Error] {response.status_code}: {response.text}")
        return None

# 构造 instruct 与 feedback 记录
def create_instruction_feedback(user_prompt, assistant_response):
    assistant_content = json.loads(assistant_response)

    # instruct 对话
    instruction_entry = {
        "type": "instruct",
        "conversations": [
            {"from": "user", "content": user_prompt},
            {"from": "assistant", "content": assistant_content}
        ]
    }

    # 模拟 tool 返回 + assistant 转自然语言
    tool_content = simulate_tool_feedback(assistant_content)
    assistant_natural = generate_feedback_natural_sentence(tool_content)

    feedback_entry = {
        "type": "feedback",
        "conversations": [
            {"from": "tool", "content": tool_content},
            {"from": "assistant", "content": {"message": assistant_natural}}
        ]
    }

    return [instruction_entry, feedback_entry]

# 主程序入口
if __name__ == "__main__":
    dataset = []
    num_samples = 100

    for _ in tqdm(range(num_samples), desc="Generating instruct+feedback pairs"):
        user_prompt = generate_user_instruction()
        if not user_prompt:
            continue
        print(f"➡️ 用户指令: {user_prompt}")
        generated_prompts.append(user_prompt)

        assistant_response = generate_assistant_response(user_prompt)
        print(f"Assistant 响应{assistant_response}")
        if not assistant_response:
            continue

        try:
            entries = create_instruction_feedback(user_prompt, assistant_response)
            dataset.extend(entries)
            # print(f"✅ 成功添加到数据集中。\n{entries}")
        except Exception as e:
            print(f"[Error] {e}")

        time.sleep(0.8)

    with open("generated_instructions.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print("✅ 数据集生成完成！保存为 generated_instructions.json")
