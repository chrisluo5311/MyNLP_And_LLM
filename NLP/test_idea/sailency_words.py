import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 1. Build a small logistic regression model
class SimpleLogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(SimpleLogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# 2. Fake sentence: Assume each word is embedded into a 5-dim vector
# (In real case, you would use GloVe, BERT embeddings, etc.)
word_embeddings = torch.tensor([
    [0.1, 0.2, 0.0, 0.4, 0.5],   # "The"
    [0.3, 0.1, 0.2, 0.0, 0.2],   # "movie"
    [0.9, 0.7, 0.8, 0.6, 0.9],   # "was"
    [1.5, 2.0, 1.8, 1.7, 2.1],   # "fantastic"
    [0.4, 0.3, 0.2, 0.5, 0.6]    # "but"
], requires_grad=True)  # << important: need gradients!

words = ["The", "movie", "was", "fantastic", "but"]

# 3. Initialize model
input_dim = word_embeddings.size(1)  # 5
model = SimpleLogisticRegression(input_dim)

# 4. Forward pass: get predictions for each word independently
outputs = model(word_embeddings)

# Average the outputs to simulate a "sentence-level" score
sentence_score = outputs.mean()

# 5. Backward pass: compute gradients
sentence_score.backward()

# 6. Now word_embeddings.grad contains saliency
saliency = word_embeddings.grad.abs()  # Take absolute value of gradients
saliency_scores = saliency.sum(dim=1)  # Sum across embedding dimensions to get 1 number per word

# 7. Normalize saliency scores for visualization
saliency_scores = saliency_scores / saliency_scores.max()
print(saliency_scores)
print(type(saliency_scores))

# 8. Visualization
plt.figure(figsize=(10, 4))  # make it wider to fit words and scores

colors = plt.cm.Reds(saliency_scores.detach().numpy())

bars = plt.bar(words, saliency_scores, color=colors)

plt.title("Word Saliency Map", fontsize=16)
# plt.axis('off')

# Add words and saliency scores under each bar
# for i, (word, score) in enumerate(zip(words, saliency_scores)):
#     plt.text(i, -0.15, f"{word}\n{score.item():.2f}", ha='center', va='top', fontsize=10)

plt.ylim(-0.5, 1.2)  # adjust so texts don't get cut off
plt.tight_layout()
plt.show()
