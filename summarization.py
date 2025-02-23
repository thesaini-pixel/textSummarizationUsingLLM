# Text Summarization Script using Facebook BART
# This script reads an input text file, summarizes its content, and saves the summary to an output file.
# It dynamically adjusts the summary length between 30% and 45% of the original text length.

# Requirements:
# Install the required libraries before running the script:
# pip install torch transformers

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import sys

# Load pre-trained summarization model and tokenizer
model_name = "facebook/bart-large-cnn"  # Using Facebook BART, optimized for text summarization
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Move model to GPU if available for faster processing
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Ensure the script receives the correct number of command-line arguments
if len(sys.argv) < 3:
    print("Usage: python script.py <input_file> <output_file>")
    sys.exit(1)

# Retrieve file names from command-line arguments
input_file = sys.argv[1]
output_file = sys.argv[2]

# Read input text from the specified file
with open(input_file, "r", encoding="utf-8") as file:
    input_text = file.read()

# Tokenize input text and determine appropriate summary length
input_tokens = tokenizer.encode(input_text, truncation=True, max_length=1024)
input_length = len(input_tokens)
min_summary_length = max(80, int(0.3 * input_length))  # At least 30% of input, minimum 80 tokens
max_summary_length = int(0.45 * input_length)  # Maximum 45% of input length

# Tokenize input and transfer it to the same device as the model
inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024).to(device)

# Generate summary with optimized settings
summary_ids = model.generate(
    **inputs,
    max_length=max_summary_length,
    min_length=min_summary_length,
    num_beams=4,  # Uses beam search to improve summary quality
    length_penalty=1.5,  # Controls verbosity of the summary
    early_stopping=True  # Ensures coherent sentence endings
)

# Decode and format the generated summary
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Save the summary to the specified output file
with open(output_file, "w", encoding="utf-8") as file:
    file.write(summary)

print("Summary saved to", output_file)
