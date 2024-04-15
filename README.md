# Liputan6 Text Summarization with BERT

This project focuses on text summarization using BERT models for Indonesian language. This project used open source dataset from
HuggingFace (source: https://huggingface.co/datasets/id_liputan6). Below are the details of the project.

## Resources:
Google Colabolatory environment with this specification:
- GPU: V100 16GB
- RAM: 12.7GB

## Used Pretrained Models:

1. [cahya/bert2bert-indonesian-summarization](https://huggingface.co/cahya/bert2bert-indonesian-summarization)
2. [Alfahluzi/bert2bert-extreme](https://huggingface.co/Alfahluzi/bert2bert-Large)

## Fine-Tuned Models:

The pretrained model `cahya/bert2bert-indonesian-summarization` was fine-tuned with different learning rates and batch sizes. Here are the fine-tuned models:

1. Learning rate 1e-5, Batch size 2
2. Learning rate 5e-5, Batch size 2
3. Learning rate 1e-5, Batch size 4
4. Learning rate 5e-5, Batch size 4

## Model Evaluation

The best performing model is fine tuned model with `learning rate 1e-5 and batch size 2` with the following Rouge scores:

- Rouge1: 0.342
- Rouge2: 0.126
- RougeL: 0.262
- RougeLSUM: 0.243

The fine-tuned model results can be accessed [here](https://huggingface.co/alfandy/bert2bert-batch2-lr1e-5-summarization) 

### All Model Results:

| Model                                          | Type                    | Rouge1 | Rouge2 | RougeL | RougeLSUM |
|------------------------------------------------|-------------------------|--------|--------|--------|-----------|
| cahya/bert2bert-indonesian-summarization      | Pretrained              | 0.312  | 0.111  | 0.252  | 0.225     |
| Alfahluzi/bert2bert-extreme                   | Pretrained              | 0.065  | 0.000  | 0.049  | 0.038     |
| model_batch_4_lr_5e-5                         | Pretrained + Fine Tuned | 0.333  | 0.120  | 0.251  | 0.235     |
| model_batch_2_lr_5e-5                         | Pretrained + Fine Tuned | 0.328  | 0.117  | 0.250  | 0.232     |
| model_batch_4_lr_1e-5                         | Pretrained + Fine Tuned | 0.320  | 0.117  | 0.245  | 0.227     |
| model_batch_2_lr_1e-5 (best model)            | Pretrained + Fine Tuned | 0.342  | 0.126  | 0.262  | 0.243     |

## Project Structure
```
├── dataset
│   ├── df_test.csv
│   ├── df_train.csv
│   ├── df_val.csv
│   ├── xtreme_dev.csv
│   └── xtreme_test.csv
├── models
│   └── model_batch_2_lr_1e-5
│       ├── config.json
│       ├── generation_config.json
│       ├── model.safetensors
│       ├── special_tokens_map
│       ├── tokenizer_config
│       ├── training_args.bin
│       └── vocab.txt
├── .gitattributes
├── .gitignore
├── app_flask.py
├── app_streamlit.py
├── text_summarization_BERT_fine_tune.ipynb
├── text_summarization_BERT_ROUGE.ipynb
└── README.md
```

- **dataset**: Directory containing dataset files.
  - **df_train.csv**: Training dataset.
  - **df_val.csv**: Validation dataset.
  - **df_test.csv**: Test dataset.
  - **xtreme_dev.csv**: Development dataset for Xtreme model.
  - **xtreme_test.csv**: Test dataset for Xtreme model.
- **models**: Directory containing trained model files.
  - **model_batch_2_lr_1e-5**: Directory for a specific fine-tuned BERT model.
    - **config.json**: Configuration file for the model.
    - **generation_config.json**: Configuration file for generation.
    - **model.safetensors**: Model file.
    - **special_tokens_map**: Special tokens map file.
    - **tokenizer_config**: Tokenizer configuration file.
    - **training_args.bin**: Training arguments file.
    - **vocab.txt**: Vocabulary file.
- **.gitattributes**: Git attributes file.
- **.gitignore**: File specifying files and directories to be ignored by Git.
- **app_flask.py**: Python file containing the Flask backend of the application.
- **app_streamlit.py**: Python file containing the Streamlit frontend of the application.
- **text_summarization_BERT_fine_tune.ipynb**: Jupyter Notebook for fine-tuning BERT models for text summarization.
- **text_summarization_BERT_ROUGE.ipynb**: Jupyter Notebook for evaluating BERT models using ROUGE metrics.
- **README.md**: Markdown file containing project description and documentation.

## Deployment
Deployment is done using Flask and Streamlit libraries.