import os
import warnings

import torch
import unsloth
from datasets import load_dataset
from transformers import BitsAndBytesConfig, DataCollatorForLanguageModeling
from trl import SFTTrainer, SFTConfig

warnings.filterwarnings("ignore")

# 设置 W&B API Key
os.environ["WANDB_API_KEY"] = ""

# 模型路径
model_path = r"D:/.cache/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# 加载模型和分词器
model, tokenizer = unsloth.FastLanguageModel.from_pretrained(
    model_path,
    max_seq_length=8192,
    dtype=torch.bfloat16,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
)

model.enable_input_require_grads()  # 梯度流优化
model.gradient_checkpointing_enable()  # 启用梯度检查点

# 加载数据集
dataset = load_dataset(
    "json",
    data_files=r"C:\Users\ADMIN\Downloads\medical_o1_sft_Chinese.json",
    split="train",
    # trust_remote_code=True
)

prompt = """Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response."""

instruction = """You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning.
Please answer the following medical question."""


def format_instruction(example):
    """格式化数据"""
    example = [
        {'role': 'system', 'content': f'{prompt}\n{instruction}'},
        {'role': 'user', 'content': example['Question']},
        {'role': 'assistant', 'content': f'{example["Complex_CoT"]}\n</think>\n{example["Response"]}'},
    ]
    text = tokenizer.apply_chat_template(
        example,
        tokenize=False,
        add_generation_prompt=False,
        padding=False,
        max_length=8192,
        chat_template=unsloth.chat_templates.qwen25_template
    )
    return {"text": text}


# 对话处理
dataset = dataset.map(
    format_instruction,
    remove_columns=dataset.column_names,
    desc='Formatted text'
)
dataset = dataset.train_test_split(test_size=0.1)

model = unsloth.FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA rank
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        # "gate_proj",
        # "up_proj",
        # "down_proj"
    ],  # 应用 LoRA 的层
    lora_alpha=8,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing=True,
)

# 内存优化训练器
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    args=SFTConfig(
        dataset_text_field="text",
        output_dir="output",
        per_device_train_batch_size=2,  # 每个设备的批次大小为1
        gradient_accumulation_steps=8,  # 累积8步等效batch_size=8
        dataset_num_proc=1,
        dataloader_num_workers=0,  # 多进程问题加载数据
        learning_rate=2e-5,  # 学习率
        num_train_epochs=3,  # 训练轮数
        logging_steps=1,
        logging_dir="logs",
        save_strategy="no",  # 关闭保存以节省显存
        # eval_strategy="epoch",
        # save_strategy="epoch",
        optim="paged_adamw_8bit",  # 8位优化器
        weight_decay=0.01,
        lr_scheduler_type="linear",  # 学习率调度器类型
        fp16=not unsloth.is_bfloat16_supported(),
        bf16=unsloth.is_bfloat16_supported(),
        max_grad_norm=0.3,  # 梯度裁剪防止溢出
        warmup_steps=5,
        seed=3407,
        packing=False,
        report_to='wandb',  # 监控平台
    ),
    # peft_config=LoraConfig(
    #     r=8,  # LoRA 秩（控制低秩近似质量）
    #     target_modules=[
    #         "q_proj",
    #         "k_proj",
    #         "v_proj",
    #         "o_proj",
    #         # "gate_proj",
    #         # "up_proj",
    #         # "down_proj"
    #     ],  # 应用 LoRA 的层
    #     lora_alpha=8,  # LoRA 权重的缩放因子
    #     lora_dropout=0,
    #     bias="none",
    #     # task_type="CAUSAL_LM"  # 因果模型
    # ),
    data_collator=DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # 关闭遮蔽语言建模
        pad_to_multiple_of=8  # 添加此项可提升GPU计算效率
    ),
)

# 开始训练
trainer_stats = trainer.train()

# 保存模型和分词器
model.save_pretrained("my_model")
tokenizer.save_pretrained("my_model")
