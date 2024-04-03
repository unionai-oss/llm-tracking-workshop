# %%
from datasets import load_dataset

dataset = load_dataset("imdb")
# %%
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
# %%
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2,
    id2label={0: "NEGATIVE", 1: "POSITIVE"},
    label2id={"NEGATIVE": 0, "POSITIVE": 1},
)


# %%
def tokenizer_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


# Use a small subset such that finetuning completes on a CPU
small_train_dataset = (
    dataset["train"].shuffle(seed=42).select(range(100)).map(tokenizer_function)
)
small_eval_dataset = (
    dataset["test"].shuffle(seed=42).select(range(50)).map(tokenizer_function)
)
# %%
import numpy as np


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": np.mean(predictions == labels)}


# %%
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="model_training",
    evaluation_strategy="epoch",
    report_to="none",
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

# %%
trainer.train()

# %%
text = "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end. Might be my favorite of the three."

# %%
from transformers import pipeline

# %%
inference = pipeline("text-classification", tokenizer=tokenizer, model=model)
# %%
inference("This movie is horrible")

# %%
inference.save_pretrained("local_hehe", safe_serialization=True)
# %%
inference2 = pipeline("text-classification", model="local_hehe")
