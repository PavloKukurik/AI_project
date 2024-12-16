import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from datasets import load_dataset
from transformers import AutoTokenizer


from simple_transformer import TransformerTranslator
import warnings
import tqdm

warnings.filterwarnings("ignore")

DEBUG = 0


class Text2EmojiDataset(Dataset):
    def __init__(self, dataset, text_tokenizer, emoji_tokenizer, max_length=64):
        self.texts = dataset["text"]
        self.emojis = dataset["emoji"]
        self.text_tokenizer = text_tokenizer
        self.emoji_tokenizer = emoji_tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text_tokens = self.text_tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        ).input_ids.squeeze()

        # Tokenize emojis

        emoji_tokens = self.emoji_tokenizer(
            self.emojis[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        ).input_ids.squeeze()

        return text_tokens, emoji_tokens


def train_transformer(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    epochs=50,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    model.to(device)
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for src, tgt in tqdm.tqdm(train_loader, position=0, leave=True):
            src = src.to(device)
            tgt = tgt.to(device)
            if DEBUG:
                print("Passed data...")
                print(src.shape, tgt.shape)

            optimizer.zero_grad()
            output = model(src, tgt)

            if DEBUG:
                print(output.view(-1, output.shape[2]).shape)
                print(tgt.view(-1).shape)

            loss = criterion(
                output.view(
                    -1,
                    output.shape[2],  # (batch_size * seq_len, vocab_size)
                ),
                tgt.view(-1),  # (batch_size * seq_len)
            )

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for src, tgt in val_loader:
                src = src.to(device)
                tgt = tgt.to(device)

                output = model(src, tgt)

                loss = criterion(output.view(-1, output.size(-1)), tgt.view(-1))
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_transformer_model.pth")


def main():
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("KomeijiForce/Text2Emoji")

    # Tokenizers
    print("Loading tokenizers...")
    text_tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
    emoji_tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")

    # Create dataset objects
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print("Splitting dataset...")
    train_dataset = Text2EmojiDataset(
        train_dataset.dataset["train"], text_tokenizer, emoji_tokenizer
    )
    val_dataset = Text2EmojiDataset(
        val_dataset.dataset["train"], text_tokenizer, emoji_tokenizer
    )

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    # Hyperparameters
    EMBED_DIM = 256
    NUM_HEADS = 4
    NUM_BLOCKS = 2
    ENCODER_VOCAB_SIZE = text_tokenizer.vocab_size
    OUTPUT_VOCAB_SIZE = emoji_tokenizer.vocab_size
    CUDA = torch.cuda.is_available()

    # Initialize model
    model = TransformerTranslator(
        embed_dim=EMBED_DIM,
        num_blocks=NUM_BLOCKS,
        num_heads=NUM_HEADS,
        encoder_vocab_size=ENCODER_VOCAB_SIZE,
        output_vocab_size=OUTPUT_VOCAB_SIZE,
        CUDA=CUDA,
    )

    params_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters: ", params_num)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=text_tokenizer.pad_token_id)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Train
    try:
        train_transformer(model, train_loader, val_loader, criterion, optimizer)
    except Exception as e:
        print(e)
        exit(0)


if __name__ == "__main__":
    main()
