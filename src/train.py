import spacy
from collections import defaultdict
from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from simple_transformer import TransformerTranslator

# -----------------------------
# Tokenizing and Dataset Prep
# -----------------------------

nlp = spacy.load("en_core_web_sm")

token_vocab = defaultdict(lambda: len(token_vocab))
emoji_vocab = defaultdict(lambda: len(emoji_vocab))
token_vocab["<pad>"]
emoji_vocab["<pad>"]


def tokenize_with_spacy(example):
    """
    Tokenize the text using spaCy and map tokens/emoji to their IDs.
    """
    tokens = [token.text for token in nlp(example["text"])]
    example["input_ids"] = [token_vocab[token] for token in tokens]

    emojis = example["emoji"]
    example["emoji_ids"] = [emoji_vocab[emoji] for emoji in emojis]
    return example


dataset = load_dataset("your_dataset_name", split="all")  # Add `split="all"` or just load it without splits

train_subset = dataset[:8000] 
valid_subset = dataset["test"][8000:10000]

subset_dataset = {"train": train_subset, "validation": valid_subset}

train_tokenized = train_subset.map(tokenize_with_spacy, batched=False)
valid_tokenized = valid_subset.map(tokenize_with_spacy, batched=False)

tokenized_dataset = {
    "train": train_tokenized,
    "validation": valid_tokenized,
}

token_vocab = dict(token_vocab)
emoji_vocab = dict(emoji_vocab)
id_to_token = {v: k for k, v in token_vocab.items()}
id_to_emoji = {v: k for k, v in emoji_vocab.items()}


class Text2EmojiDataset(Dataset):
    def __init__(self, dataset, max_seq_len):
        self.dataset = dataset
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        input_ids = item["input_ids"] + [token_vocab["<pad>"]] * (
            self.max_seq_len - len(item["input_ids"])
        )
        emoji_ids = item["emoji_ids"] + [emoji_vocab["<pad>"]] * (
            self.max_seq_len - len(item["emoji_ids"])
        )

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(emoji_ids, dtype=torch.long),
        }


# Hyperparameters
MAX_SEQ_LEN = 50
BATCH_SIZE = 32
EMBED_DIM = 128
NUM_BLOCKS = 4
NUM_HEADS = 8
EPOCHS = 10
LR = 0.001

train_dataset = Text2EmojiDataset(tokenized_dataset["train"], max_seq_len=MAX_SEQ_LEN)
valid_dataset = Text2EmojiDataset(
    tokenized_dataset["validation"], max_seq_len=MAX_SEQ_LEN
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)

# -----------------------------
# Transformer Model
# -----------------------------

model = TransformerTranslator(
    embed_dim=EMBED_DIM,
    num_blocks=NUM_BLOCKS,
    num_heads=NUM_HEADS,
    encoder_vocab_size=len(token_vocab),
    output_vocab_size=len(emoji_vocab),
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# -----------------------------
# Training Loop
# -----------------------------

optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss(ignore_index=token_vocab["<pad>"])


def train_one_epoch(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch in data_loader:
        src = batch["input_ids"].to(device)
        tgt = batch["labels"].to(device)

        optimizer.zero_grad()
        output = model(src, tgt[:, :-1])
        loss = criterion(output.reshape(-1, output.shape[-1]), tgt[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)


def validate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in data_loader:
            src = batch["input_ids"].to(device)
            tgt = batch["labels"].to(device)

            output = model(src, tgt[:, :-1])
            loss = criterion(
                output.reshape(-1, output.shape[-1]), tgt[:, 1:].reshape(-1)
            )
            total_loss += loss.item()

    return total_loss / len(data_loader)


for epoch in range(EPOCHS):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss = validate(model, valid_loader, criterion, device)

    print(
        f"Epoch {epoch + 1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
    )
