import json

# 将自定义格式的数据转换为 Qwen 可接受的 ChatML 格式
def convert_to_qwen_chatml(data):
    qwen_samples = []
    for item in data:
        if item["type"] == "instruct":
            user_msg = item["conversations"][0]["content"]
            assistant = item["conversations"][1]["content"]
            tool_json = json.dumps({
                "type": assistant.get("type", "unknown"),
                "name": assistant.get("name", "unknown"),
                "arguments": assistant.get("arguments", {})
            }, ensure_ascii=False)
            response_msg = assistant.get("message", "") + "\n" + tool_json
            chatml = f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n{response_msg}<|im_end|>"
            qwen_samples.append({"text": chatml})

        elif item["type"] == "feedback":
            tool_resp_obj = item["conversations"][0]["content"]
            # 如果是 dict 则提取 status 字段
            tool_resp = tool_resp_obj.get("status") if isinstance(tool_resp_obj, dict) else tool_resp_obj
            assistant_msg = item["conversations"][1]["content"].get("message", "")
            chatml = f"<|im_start|>tool\n{tool_resp}<|im_end|>\n<|im_start|>assistant\n{assistant_msg}<|im_end|>"
            qwen_samples.append({"text": chatml})

    return qwen_samples

if __name__ == "__main__":
    with open("generated_instructions.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    converted = convert_to_qwen_chatml(data)

    with open("qwen_chatml.json", "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    print("✅ 转换完成，保存为 qwen_chatml.json")