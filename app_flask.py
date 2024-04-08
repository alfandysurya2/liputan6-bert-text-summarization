from flask import Flask, request, jsonify
from transformers import BertTokenizer, EncoderDecoderModel
import torch
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

app = Flask(__name__)

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained("alfandy/bert2bert-batch2-lr1e-5-summarization")
model = EncoderDecoderModel.from_pretrained("alfandy/bert2bert-batch2-lr1e-5-summarization")

@app.route("/summarize", methods=["POST"])
def summarize():
    # Get input text from request
    input_text = request.json.get("text", "")

    # Tokenize input text
    inputs = tokenizer(input_text, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    # Generate summary
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        min_length=20,
        max_length=80, 
        num_beams=10,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True,
        no_repeat_ngram_size=2,
        use_cache=True,
        do_sample=True,
        temperature=0.8,
        top_k=50,
        top_p=0.95
    )

    # Decode the generated summary
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)  # Assuming batch size 1

    # Return the generated summary
    return jsonify({"summary": summary})

if __name__ == "__main__":
    app.run(debug=True)