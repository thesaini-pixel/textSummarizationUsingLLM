import requests
import json
import sys

def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def save_text_file(text, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(text)

def summarize_text(text, summary_ratio=0.4):
    # Hugging Face Inference API endpoint (Falcon 7B or other models)
    API_URL = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"
    headers = {"Authorization": f"Bearer you-api-key"}ß
    
    # Reduce input size to prevent exceeding model token limits
    text = ' '.join(text.split()[:1024])
    prompt = f"Summarize the following text to {int(len(text.split()) * summary_ratio)} words:\n{text}"
    
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 300, "temperature": 0.7, "do_sample": True}}
    response = requests.post(API_URL, headers=headers, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        return result[0]["generated_text"] if isinstance(result, list) else result.get("generated_text", "")
    else:
        return f"Error: {response.status_code}, {response.text}"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python summarization.py <input_text_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]  # Get input file name from command line argument
    output_file = 'summary.txt'  # Path to save summary

    # Read the input text
    article_text = read_text_file(input_file)

    # Summarize the text using the API
    summary = summarize_text(article_text, summary_ratio=0.4)

    # Save the summary to a file
    save_text_file(summary, output_file)

    print(f"Summary saved to {output_file}")
