from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
import torch


class GPT2:
    id = 1

    def __init__(self, modelName=None) -> None:
        if modelName:
            self.name = modelName
        else:
            self.name = f"gpt2_{GPT2.id}"
            GPT2.id += 1

        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.dataloader = DataLoader

    def init_data_loader(self, data, max_length=280, batch_size=32, shuffle=True, pin_memory=True, num_workers=0):
        self.dataset = __TweetDataset(
            data, tokenizer=self.tokenizer, max_length=max_length)

        self.dataloader = DataLoader(
            self.dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory, num_workers=num_workers)

    def init_optimizer(self, lr=1e-5, num_warmup_steps=100, num_training_steps=1000):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train(self, epochs=1, max_grad_norm=1.0):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        self.model.train()
        epoch_loss = 0.0

        for epoch in range(epochs):
            for batch in self.dataloader:
                input_ids = batch["input_ids"].to(self.device)

                # forward pass
                outputs = self.model(input_ids, labels=input_ids)
                loss = outputs.loss
                total_loss += loss.item()

                # backward pass
                loss.backward()

                # Apply gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_grad_norm)

                # update learning rate
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            epoch_loss = 0

            print(
                f"Epoch {epoch + 1}/{epochs}, Average Loss: {epoch_loss / len(self.dataloader)}")
            torch.save(self.model.state_dict(),
                       f"{self.name}{epoch + 1}.pth")

    def save_model(self, path,):
        self.model.save_pretrained(path)

    def generate_text(self, prompt=None):
        if prompt:
            generated_tweets = self.model.generate(prompt,
                                                   max_length=100, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95)
        else:
            generated_tweets = self.model.generate(
                max_length=100, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95)

        decoded_tweets = self.tokenizer.decode(
            generated_tweets[0], skip_special_tokens=True)
        return decoded_tweets


class __TweetDataset(Dataset):
    def __init__(self, tweets, tokenizer, max_length=280):
        self.encodings = tokenizer(
            tweets, truncation=True, max_length=max_length, padding="max_length")

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
