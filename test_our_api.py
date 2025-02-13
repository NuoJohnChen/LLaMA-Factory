




paper_content="""
In this paper, we propose a metric-based probing method, namely, CAT-probing, to quantitatively evaluate how CodePTMs Attention scores relate to distances between AST nodes. First, to denoise the input code sequence in the original attention scores matrix, we classify the rows/cols by token types that are pre-defined by compilers, and then retain tokens whose types have the highest proportion scores to derive a filtered attention matrix (see Figure 1(b)). Meanwhile, inspired by the works (Wang et al., 2020; Zhu et al., 2022), we add edges to improve the connectivity of AST and calculate the distances between nodes corresponding to the selected tokens, which generates a distance matrix as shown in Figure 1(c). After that, we define CAT-score to measure the matching degree between the filtered attention matrix and the distance matrix. Specifically, the point-wise elements of the two matrices are matched if both the two conditions are satisfied: 1) the attention score is larger than a threshold; 2) the distance value is smaller than a threshold. If only one condition is reached, the elements are unmatched. We calculate the CAT-score by the ratio of the number of matched elements to the summation of matched and unmatched elements. Finally, the CAT-score is used to interpret how CodePTMs attend code structure, where a higher score indicates that the model has learned more structural information.

Our main contributions can be summarized as follows:
• We propose a novel metric-based probing method CAT-probing to quantitatively interpret how CodePTMs attend code structure.
• We apply CAT-probing to several representative CodePTMs and perform extensive experiments to demonstrate the effectiveness of our method (See Section 4.3).
• We draw two fascinating observations from the empirical evaluation: 1) The token types that PTMs focus on vary with programming languages and are quite different from the general perceptions of human programmers (See Section 4.2). 2) The ability of CodePTMs to capture code structure dramatically differs with layers (See Section 4.4).
"""

selected_content="""
After that, we define CAT-score to measure the matching degree between the filtered attention matrix and the distance matrix.
"""

question="""
help me redefine cat-score based on the context.
[Requirements]
1. Maintain consistency with the methodology described in the paper
2. Use precise mathematical notation if applicable
3. Keep the definition concise (1-2 sentences)
4. Explicitly mention the two threshold conditions
"""


import requests
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8088/v1",
    api_key="sk-1234567890"
)

content = f"""
Please improve the selected content based on the following. Act as an expert model for improving articles **PAPER_CONTENT**.\n
The output needs to answer the **QUESTION** on **SELECTED_CONTENT** in the input. Avoid adding unnecessary length, unrelated details, overclaims, or vague statements. Focus on clear, concise, and evidence-based improvements that align with the overall context of the paper.\n

<PAPER_CONTENT>
{paper_content}
</PAPER_CONTENT>\n

<SELECTED_CONTENT>
{selected_content}
</SELECTED_CONTENT>\n

<QUESTION>
{question}
</QUESTION>\n
"""

try:
    response = client.chat.completions.create(
        model="xtragpt",
        messages=[{"role": "user", "content": content}],
        temperature=0.7,
        max_tokens=16384 # 确保不超过启动时设置的--max-model-len 42447
    )
    print(response.choices[0].message.content)
except Exception as e:
    print(f"API Error: {str(e)}")