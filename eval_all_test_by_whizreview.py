from vllm import LLM, SamplingParams
import torch
import re
from transformers import AutoTokenizer
import transformers
import os
import json
from transformers import pipeline
import sys
from tqdm import tqdm

def process_text(text, skip_appendix=True):
    pattern = re.compile(r"Under review as a conference paper at ICLR 2024", re.IGNORECASE)
    text = pattern.sub("", text)

    pattern = re.compile(r"Published as a conference paper at ICLR 2024", re.IGNORECASE)
    text = pattern.sub("", text)
    
    if skip_appendix:
        match = re.search(r"REFERENCES", text, re.IGNORECASE)

        if match:
            # Truncate the text at "REFERENCES"
            text = text[:match.start()]

    match = re.search(r"ABSTRACT", text, re.IGNORECASE)

    if match:
        text = text[match.start():]

    return text.strip()

model_id = "/shared/hdd/nuochen/models/WhizReviewer-ML-sft_merge"#"/ssd1/models/WhizReviewer-ML-Llama3.1-8B"
pipeline = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

# 确保在创建pipeline后立即设置pad_token_id
if pipeline.tokenizer.pad_token_id is None:
    pipeline.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    pipeline.model.resize_token_embeddings(len(pipeline.tokenizer))
    pipeline.tokenizer.pad_token_id = pipeline.tokenizer.convert_tokens_to_ids('[PAD]')

print(f"Pad token ID: {pipeline.tokenizer.pad_token_id}")
print(f"Pad token: {pipeline.tokenizer.pad_token}")

# system_prompt = \
# """You are an expert academic reviewer tasked with providing a balanced evaluation of research papers. Provide only the scores for the following aspects, in the format described below:

# 1. Soundness
# 2. Presentation
# 3. Contribution
# 4. Rating (1-10)
# 5. Decision (Accept/Reject)

# Output format ONLY:
# ## Reviewer\n\n### 1. Soundness: [score]\n2. Presentation: [score]\n3. Contribution: [score]\n4. Rating: [score]\n5. Decision: Accept or Reject

# Replace [score] with actual numeric values. For the Decision, output either "Accept" or "Reject" based on your evaluation. Do not include any explanations, justifications, or reviews. Only the scores and the specified output format are required.
# """
system_prompt = \
"""You are an expert academic reviewer tasked with providing a thorough and balanced evaluation of research papers.
"""


a1= """
## Review Form
Below is a description of the questions you will be asked on the review form for each paper and some guidelines on what to consider when answering these questions.
When writing your review, please keep in mind that after decisions have been made, reviews and meta-reviews of accepted papers and opted-in rejected papers will be made public. 

1. Soundness: Please assign the paper a numerical rating on the following scale to indicate the soundness of the technical claims, experimental and research methodology and on whether the central claims of the paper are adequately supported with evidence.
  4: excellent
  3: good
  2: fair
  1: poor

2. Presentation: Please assign the paper a numerical rating on the following scale to indicate the quality of the presentation. This should take into account the writing style and clarity, as well as contextualization relative to prior work.
  4: excellent
  3: good
  2: fair
  1: poor

3. Contribution: Please assign the paper a numerical rating on the following scale to indicate the quality of the overall contribution this paper makes to the research area being studied. Are the questions being asked important? Does the paper bring a significant originality of ideas and/or execution? Are the results valuable to share with the broader NeurIPS community.
  4: excellent
  3: good
  2: fair
  1: poor

4. Overall: Please provide an "overall score" for this submission. Choices: 
  10: Award quality: Technically flawless paper with groundbreaking impact on one or more areas of AI, with exceptionally strong evaluation, reproducibility, and resources, and no unaddressed ethical considerations.
  9: Very Strong Accept: Technically flawless paper with groundbreaking impact on at least one area of AI and excellent impact on multiple areas of AI, with flawless evaluation, resources, and reproducibility, and no unaddressed ethical considerations.
  8: Strong Accept: Technically strong paper with, with novel ideas, excellent impact on at least one area of AI or high-to-excellent impact on multiple areas of AI, with excellent evaluation, resources, and reproducibility, and no unaddressed ethical considerations.
  7: Accept: Technically solid paper, with high impact on at least one sub-area of AI or moderate-to-high impact on more than one area of AI, with good-to-excellent evaluation, resources, reproducibility, and no unaddressed ethical considerations.
  6: Weak Accept: Technically solid, moderate-to-high impact paper, with no major concerns with respect to evaluation, resources, reproducibility, ethical considerations.
  5: Borderline accept: Technically solid paper where reasons to accept outweigh reasons to reject, e.g., limited evaluation. Please use sparingly.
  4: Borderline reject: Technically solid paper where reasons to reject, e.g., limited evaluation, outweigh reasons to accept, e.g., good evaluation. Please use sparingly.
  3: Reject: For instance, a paper with technical flaws, weak evaluation, inadequate reproducibility and incompletely addressed ethical considerations.
  2: Strong Reject: For instance, a paper with major technical flaws, and/or poor evaluation, limited impact, poor reproducibility and mostly unaddressed ethical considerations.
  1: Very Strong Reject: For instance, a paper with trivial results or unaddressed ethical considerations
"""
a2= """THOUGHT:
<THOUGHT>

REVIEW JSON:
```json
<JSON>
```

```Rating
<Overall>
```


In <THOUGHT>, first briefly discuss your intuitions and reasoning for the evaluation.
Detail your high-level arguments, necessary choices and desired outcomes of the review.
Do not make generic comments here, but be specific to your current paper.
Treat this as the note-taking phase of your review.

In <JSON>, provide the review in JSON format with the following fields in the order:
- "Summary": A summary of the paper content and its contributions.
- "Strengths": A list of strengths of the paper.
- "Weaknesses": A list of weaknesses of the paper.
- "Originality": A rating from 1 to 4 (low, medium, high, very high).
- "Quality": A rating from 1 to 4 (low, medium, high, very high).
- "Clarity": A rating from 1 to 4 (low, medium, high, very high).
- "Significance": A rating from 1 to 4 (low, medium, high, very high).
- "Questions": A set of clarifying questions to be answered by the paper authors.
- "Limitations": A set of limitations and potential negative societal impacts of the work.
- "Ethical Concerns": A boolean value indicating whether there are ethical concerns.
- "Soundness": A rating from 1 to 4 (poor, fair, good, excellent).
- "Presentation": A rating from 1 to 4 (poor, fair, good, excellent).
- "Contribution": A rating from 1 to 4 (poor, fair, good, excellent).
- "Overall"(<Overall>): A rating from 1 to 10 (very strong reject to award quality).
- "Confidence": A rating from 1 to 5 (low, medium, high, very high, absolute).
- "Decision": A decision that has to be one of the following: Accept, Reject.

For the "Decision" field, don't use Weak Accept, Borderline Accept, Borderline Reject, or Strong Reject. Instead, only use Accept or Reject.
This JSON will be automatically parsed, so ensure the format is precise.
"""



a3= "Here is the paper you are asked to review:"

# 读取预测文件
prediction_file = "/shared/ssd/ConferenceQA_rating/rating_test_prediction_avg.jsonl"
with open(prediction_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 创建数据集并存储在列表中
def generate_prompts():
    for line in tqdm(lines, desc="Processing lines"):
        data = json.loads(line)
        markdown_context = process_text(data['paper_content'], skip_appendix=True)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": a1+a2+a3+markdown_context},
        ]
        yield messages

prompts = list(tqdm(generate_prompts(), desc="Processing lines"))

# 使用Dataset进行批处理
batch_size = 1 # 可以根据GPU内存调整批处理大小
num_batches = len(prompts) // batch_size + (1 if len(prompts) % batch_size != 0 else 0)

all_outputs = []
for i in tqdm(range(num_batches), desc="Generating outputs"):
    batch_prompts = prompts[i * batch_size:(i + 1) * batch_size]
    outputs = pipeline(
        batch_prompts,
        max_new_tokens=64
    )
    # print(f"outputs type: {type(outputs)}")
    # print(f"outputs content: {outputs}")
    # print(f"outputs[0] type: {type(outputs[0])}")
    # print(f"outputs[0] content: {outputs[0]}")
    # print(f"outputs[0]['generated_text'] type: {type(outputs[0]['generated_text'])}")
    # print(f"outputs[0]['generated_text'] content: {outputs[0]['generated_text']}")
    # print(f"outputs[0]['generated_text'][-1] type: {type(outputs[0]['generated_text'][-1])}")
    # print(f"outputs[0]['generated_text'][-1] content: {outputs[0]['generated_text'][-1]}")
    review_output = outputs[0][0]["generated_text"][-1]["content"]
    print(f"模型输出内容: {review_output}")
    all_outputs.append(review_output)

updated_lines = []
for i, line in enumerate(tqdm(all_outputs, desc="Updating lines")):
    try:
        data = json.loads(line)
        review_output = all_outputs[i]
        ### 1. Soundness: {soundness}\n2. Presentation: {presentation}\n3. Contribution: {contribution}\n4. Rating: {rating}\n5. Decision: {decision}
        pattern = r"### 1\. Soundness: (\d+(\.\d+)?)\n2\. Presentation: (\d+(\.\d+)?)\n3\. Contribution: (\d+(\.\d+)?)\n4\. Rating: (\d+(\.\d+)?)\n5\. Decision: (Accept|Reject)"
        match = re.search(pattern, review_output)
        
        if match:
            data.update({
                'predict_soundness': float(match.group(1)),  # 完整的Soundness评分
                'predict_presentation': float(match.group(3)),  # 完整的Presentation评分
                'predict_contribution': float(match.group(5)),  # 完整的Contribution评分
                'predict_rating': float(match.group(7)),        # 完整的Rating评分
                'predict_decision': match.group(9)              # Decision
            })
        else:
            print(f"未能提取到分数信息: {data['article']}")
            print(f"模型输出内容: {review_output}")
            sys.exit(1)  # 终止程序
        
        updated_lines.append(json.dumps(data, ensure_ascii=False) + '\n')
    except Exception as e:
        print(f"Error processing line {i}: {e}")
        print(type(line))
        sys.exit(1)  # 终止程序

# 将更新后的内容写回文件
with open(prediction_file, 'w', encoding='utf-8') as f:
    f.writelines(updated_lines)

print(f"预测完成，结果已更新到文件: {prediction_file}")