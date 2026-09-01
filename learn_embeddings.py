import os
from sentence_transformers import SentenceTransformer

# SECURE WAY: Load token from your local environment instead of hardcoding
# Run 'export HF_TOKEN="your_actual_token"' in your WSL terminal before running the script
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "fallback_token_if_needed")

# 1. Load a pretrained Sentence Transformer model
# (Note: all-MiniLM-L6-v2 is public, so HF_TOKEN is actually optional here!)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# The sentences to encode
sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium.",
]

# 2. Calculate embeddings by calling model.encode()
embeddings = model.encode(sentences)

print("Shape:", embeddings.shape)          # Expected output: (3, 384)
print("Type:", type(embeddings))            # Expected output: <class 'numpy.ndarray'>
print("First 5 values of matrix:\n", embeddings[:2, :5])  # Slice a small chunk to look at

# 3. Calculate the embedding similarities
similarities = model.similarity(embeddings, embeddings)
print("\nSimilarity Matrix:\n", similarities)
# Expected output: A 3x3 PyTorch tensor showing cosine similarities
