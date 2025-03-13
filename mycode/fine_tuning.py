import os

import markdownify
import unsloth
from accelerate import Accelerator
from bs4 import BeautifulSoup
from datasets import load_dataset
from peft import LoraConfig
from transformers import DataCollatorForLanguageModeling, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from unsloth_zoo.dataset_utils import train_on_responses_only

# 设置 W&B API Key
os.environ["WANDB_API_KEY"] = "982e615d9587b87d31dcf56f7d80e2a859f81ab3"

# 定义加载模型的配置
max_seq_length = 8192
dtype = None  # 自动选择最佳数据类型
load_in_4bit = True  # 启用4位量化以减少内存使用

dataset_path = r"/root/python_project/StoryFusion/spider/processed_data.jsonl"  # 数据集路径
# dataset_path = r"/root/python_project/StoryFusion/spider/processed_dataset"  # 数据集路径
model_path = r"/mnt/models/deepseek-ai/DeepSeek-R1-Distill-Llama-8B"  # 模型路径

# 初始化 Accelerator
accelerator = Accelerator()
model, tokenizer = unsloth.FastLanguageModel.from_pretrained(
    model_path,
    max_seq_length=max_seq_length,
    dtype=dtype,
    # load_in_4bit=load_in_4bit,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
    )
)

# 将模型和分词器分配到多卡
model, tokenizer = accelerator.prepare(model, tokenizer)
# tokenizer.pad_token = tokenizer.eos_token

# model.enable_input_require_grads()  # 梯度流优化
model.gradient_checkpointing_enable()  # 启用梯度检查点

# 为分词器应用聊天模板
tokenizer = unsloth.get_chat_template(
    tokenizer,
    chat_template="llama-3.1",
)

# 加载数据集
dataset = load_dataset(
    'json',
    data_files=dataset_path,
    split="train"
)


def formatting_prompts_func(examples):
    """
    格式化对话数据为文本
    :param examples: 对话数据（批）
    :return: 模板数据（批）
    """
    convos = examples["content"]

    texts = []
    for convo in convos:
        # 解析HTML
        soup = BeautifulSoup(convo, 'html.parser')
        # 移除所有的图片标签
        for img in soup.find_all('img'):
            img.decompose()
        # 移除所有的section标签
        for section in soup.find_all('section'):
            section.decompose()
        # 将处理后的HTML转换为Markdown
        markdown_convo = markdownify.markdownify(str(soup))

        # 假设markdown_convo是一个字符串，需要将其转换为对话消息格式
        # 这里假设对话消息格式为 [{"role": "user", "content": "user message"}, {"role": "assistant", "content": "assistant message"}]
        # 你需要根据实际情况调整这个部分
        messages = [
            {"role": "user", "content": markdown_convo},  # 假设markdown_convo是用户的消息
            {"role": "assistant", "content": markdown_convo}  # 假设没有助手的消息，或者你可以根据实际情况添加
        ]
        # 使用分词器的聊天模板格式化对话数据
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)
    return {"text": texts}


# 格式化对话数据
dataset = dataset.map(
    formatting_prompts_func,
    batch_size=10,
    batched=True,
    remove_columns=list(dataset.column_names)
)

# 定义训练配置
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    data_collator=DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # 关闭遮蔽语言建模
        pad_to_multiple_of=8,  # 提升GPU计算效率
    ),
    peft_config=LoraConfig(
        r=16,  # LoRA秩（控制低秩近似质量）
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            # "gate_proj",
            # "up_proj",
            # "down_proj"
        ],  # 应用LoRA的层
        lora_alpha=16,  # LoRA权重的缩放因子
        lora_dropout=0,
        bias="none",
        use_rslora=False,
    ),
    args=SFTConfig(
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataloader_num_workers=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-5,
        num_train_epochs=3,
        optim="paged_adamw_8bit",
        fp16=not unsloth.is_bfloat16_supported(),
        bf16=unsloth.is_bfloat16_supported(),
        logging_steps=1,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        logging_dir="logs",
        report_to='wandb'
    ),
)

# 只训练回复部分
trainer = train_on_responses_only(
    trainer,
    # instruction_part="<|im_start|>user\n",
    # response_part="<|im_start|>assistant\n",
    instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
    response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
)

# 开始训练
trainer_stats = trainer.train()

# 保存模型和分词器
model.save_pretrained("MyModel")
tokenizer.save_pretrained("MyModel")
