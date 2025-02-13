from llama_factory import LlamaFactory  # Assuming this is how you import the package

# Instantiate the model
model = LlamaFactory(model_name="your-model-name")

# Prepare your input data (a list of dictionaries similar to your example)
inputs = [
    {
        "instruction": "Please improve the selected content based on the following.",
        "input": "Act as an expert model for improving articles **PAPER_CONTENT**...",
        "output": "Improved content...",
        "section": "conclusion",
        "criteria": "Clarity and Impact of Key Innovations and Findings"
    },
    # Add more items for batch inference
]

# Define a function for batch processing
def batch_inference(model, inputs):
    results = []
    for item in inputs:
        instruction = item["instruction"]
        input_text = item["input"]
        # Call the model to get the output based on the instruction and input
        output = model.inference(instruction, input_text)
        results.append(output)
    return results

# Run the batch inference
batch_results = batch_inference(model, inputs)

# Print or process the results
for result in batch_results:
    print(result)
