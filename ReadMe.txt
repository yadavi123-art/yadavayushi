Offline Chat-Reply Recommendation System (GPT-2)
====================================================

This project fine-tunes a GPT-2 model offline to predict the next reply from User A
given User B's latest message and recent history. It follows the Round 4 – AI–ML
Developer Intern specifications.

Folder Structure
----------------
[your_meetmux_email_id]/
├─ ChatRec_Model.ipynb            # Training + Evaluation + Export
├─ Model.joblib                   # Model metadata (path, special tokens, notes)
├─ Report.pdf                     # Summary report
├─ ReadMe.txt                     # This file
└─ (generated at runtime)
   ├─ gpt2_offline_chatbot/
   │  ├─ checkpoints/            # Trainer checkpoints
   │  └─ artifact/               # Final model + tokenizer + metrics + samples
   └─ offline_chatbot/
      └─ processed_conversations.csv  # Built from raw chats

Prerequisites (Offline)
-----------------------
- Python 3.10+
- Preloaded libraries: transformers, torch, numpy, pandas, scikit-learn, nltk, joblib, matplotlib
- Preloaded Hugging Face weights for gpt2 (no internet access during training/inference)
- Dataset: two-person conversation CSV with columns:
  [Conversation ID, Timestamp, Sender, Message]

Data Preparation
----------------
1) Place your raw conversation CSV at:
   /mnt/data/conversationfile.xlsx - userAuserB.csv
2) Run the "Preprocessing" cell in the notebook or use the provided script
   to generate:
   /mnt/data/offline_chatbot/processed_conversations.csv

Training (ChatRec_Model.ipynb)
------------------------------
1) Open and run all cells in ChatRec_Model.ipynb.
2) What the notebook does:
   - Loads processed_conversations.csv
   - Adds special tokens: <BOS>, <EOS>, <SEP>, <USER_A>, <USER_B>
   - Fine-tunes GPT-2 with loss masked to the reply portion
   - Evaluates Perplexity, BLEU, ROUGE-L
   - Saves artifacts to ./gpt2_offline_chatbot/artifact/

Key Hyperparameters
-------------------
- Model: gpt2 (causal LM)
- Epochs: 3
- Learning rate: 5e-5
- Batch size: 2 (per device)
- Max sequence length: 512
- Special tokens: <BOS>, <EOS>, <SEP>, <USER_A>, <USER_B>
- Context window: last ~6 messages (configurable in preprocessing)

Outputs
-------
- ./gpt2_offline_chatbot/artifact/
  - config.json, pytorch_model.bin, tokenizer files
  - metrics.json (val_bleu_mean, val_rougeL_mean, val_perplexity)
  - samples.jsonl (validation examples: context, reference, hypothesis)
  - Model.joblib (also provided at project root)

Running Inference (Offline)
---------------------------
Use the helper function from the notebook. Example (standalone Python):

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ARTIFACT = "./gpt2_offline_chatbot/artifact"
tokenizer = AutoTokenizer.from_pretrained(ARTIFACT)
model = AutoModelForCausalLM.from_pretrained(ARTIFACT)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def build_prompt(context: str) -> str:
    ctx = context.replace("User A:", "<USER_A>:").replace("User B:", "<USER_B>:")
    return f"<BOS> {ctx}\n<USER_A>: "

@torch.no_grad()
def generate_reply(context, max_new_tokens=64, temperature=0.7, top_p=0.9):
    prompt = build_prompt(context)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text.split("<USER_A>:")[-1].strip()

context = "User B: Any plans for Saturday?"
print("Predicted reply:", generate_reply(context))
```

Troubleshooting
---------------
- If tokenizer complains about missing special tokens, re-run the training cell that adds them.
- If memory is limited, reduce max_length to 384, batch size to 1, or enable gradient_accumulation_steps.
- If validation BLEU/ROUGE are low, increase epochs, expand the context window, or add more data.

Submission Checklist
--------------------
- [ ] ChatRec_Model.ipynb executes fully offline
- [ ] Model.joblib present at project root
- [ ] Report.pdf included
- [ ] ReadMe.txt included
- [ ] ./gpt2_offline_chatbot/artifact/ created after training
- [ ] processed_conversations.csv saved

Notes
-----
- BERT is encoder-only (not generative); GPT-2 is recommended for generation.
- T5 also works (seq2seq), but is larger and can be slower on CPU-only setups.
- All steps honor the offline constraint from the problem statement.
