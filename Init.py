import torch
from sentence_transformers import SentenceTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"

model_id = "google/embeddinggemma-300M"
model = SentenceTransformer(model_id).to(device=device)

print(f"Device: {model.device}")
print(model)
print("Total number of parameters in the model:", sum([p.numel() for _, p in model.named_parameters()]))

words = ["apple", "banana", "car"]

# Calculate embeddings by calling model.encode()
embeddings = model.encode(words)

print(embeddings)
for idx, embedding in enumerate(embeddings):
  print(f"Embedding {idx+1} (shape): {embedding.shape}")