import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import json
from torch.cuda.amp import autocast, GradScaler

# 超参数配置
MODEL_PATH = "./qwen"  # 你的模型权重路径
DATA_PATH = "./qwen/qwen_chatml.json"  # 你的训练数据路径
OUTPUT_DIR = "./outputs"  # 保存微调后的模型
MAX_LENGTH = 2048  # 输入序列最大长度

# 自定义数据集
class ChatMLDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=2048):
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        self.tokenizer = tokenizer
        self.samples = raw_data
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]["text"]
        tokenized = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": tokenized.input_ids.squeeze(0),
            "attention_mask": tokenized.attention_mask.squeeze(0),
            "labels": tokenized.input_ids.squeeze(0),
        }

# 加载模型和分词器
print(f"加载分词器: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

print(f"加载模型: {MODEL_PATH}")
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, 
                                             trust_remote_code=True, 
                                             torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                                             device_map="auto")

# 加载数据集
print(f"# 加载数据集: {DATA_PATH}")
dataset = ChatMLDataset(DATA_PATH, tokenizer, max_length=MAX_LENGTH)

# 训练参数
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,
    num_train_epochs=3,
    per_device_train_batch_size=1,  # 减小批次大小
    gradient_accumulation_steps=16,  # 增加梯度累积步骤
    logging_steps=10,
    learning_rate=2e-4,
    warmup_steps=50,
    weight_decay=0.01,
    # fp16=torch.cuda.is_available(),  # 使用 fp16 训练
    fp16=False,
    max_grad_norm=0.5,  # 梯度裁剪
    optim="adamw_torch",
    report_to="none"
)

# 初始化 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

# 创建 GradScaler 实例
scaler = GradScaler()

# 开始训练
if __name__ == "__main__":
    print("开始训练...")
    
    # # 修改为手动控制训练步骤
    # for epoch in range(training_args.num_train_epochs):
    #     model.train()
    #     for step, batch in enumerate(dataset):
    #         # 将输入数据传输到 GPU
    #         batch = {key: value.to(model.device) for key, value in batch.items()}
            
            

    #         # 确保 causal_mask 和 attention_mask 的维度正确（四维）
    #         if "causal_mask" in batch:
    #             print(f"causal_mask shape before adjustment: {batch['causal_mask'].shape}")
    #             # 如果 causal_mask 维度大于 4，使用 squeeze() 减少多余维度
    #             if batch['causal_mask'].dim() > 4:
    #                 batch['causal_mask'] = batch['causal_mask'].squeeze(0)
    #             if batch['causal_mask'].dim() == 1:  # 如果 causal_mask 是一维张量
    #                 batch['causal_mask'] = batch['causal_mask'].unsqueeze(0).unsqueeze(0)  # 转为四维 (1, 1, seq_len)
    #             print(f"causal_mask shape after adjustment: {batch['causal_mask'].shape}")

    #         if "attention_mask" in batch:
    #             print(f"attention_mask shape before adjustment: {batch['attention_mask'].shape}")
    #             if batch['attention_mask'].dim() == 1:  # 如果 attention_mask 是一维张量
    #                 batch['attention_mask'] = batch['attention_mask'].unsqueeze(0).unsqueeze(0)  # 转为四维 (1, 1, seq_len)
    #             print(f"attention_mask shape after adjustment: {batch['attention_mask'].shape}")
            
    #         # 清空梯度
    #         model.zero_grad()
            
    #         # 使用 autocast 进行混合精度训练
    #         with autocast(enabled=True):
    #             outputs = model(**batch)
    #             loss = outputs.loss

    #         # 使用 GradScaler 来缩放损失
    #         scaler.scale(loss).backward()
    #         scaler.step(trainer.optimizer)
    #         scaler.update()

    #         if step % training_args.logging_steps == 0:
    #             print(f"Epoch {epoch+1}/{training_args.num_train_epochs} | Step {step}/{len(dataset)} | Loss: {loss.item()}")

    # # 保存训练后的模型
    # trainer.save_model(OUTPUT_DIR)
    # tokenizer.save_pretrained(OUTPUT_DIR)
    # print("✅ 训练完成！模型保存到:", OUTPUT_DIR)

    print("开始训练...")
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("✅ 训练完成！模型保存到:", OUTPUT_DIR)
