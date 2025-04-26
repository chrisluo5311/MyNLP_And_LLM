import torch
import matplotlib.pyplot as plt

# 1. Create 3 fake word vectors (pretend they are from GloVe)
# Each word is 3-dimensional for easy visualization
cat_vec = torch.tensor([0.2, 0.1, 0.4])
dog_vec = torch.tensor([0.3, 0.0, 0.5])
pet_vec = torch.tensor([0.25, 0.05, 0.45])

# Stack them together
word_vectors = torch.stack([cat_vec, dog_vec, pet_vec])  # Shape (3 words, 3 dimensions)

print("Original word vectors:")
print(word_vectors)

# 2. Take average over words (dimension 0)
sentence_vector = word_vectors.mean(dim=0)

print("\nAveraged sentence vector:")
print(sentence_vector)

# 3. Plot before and after
# Before: plot all word vectors individually
# After: plot the averaged sentence vector

fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Plot each word vector
axs[0].plot([1, 2, 3], cat_vec, marker='o', label='cat')
axs[0].plot([1, 2, 3], dog_vec, marker='o', label='dog')
axs[0].plot([1, 2, 3], pet_vec, marker='o', label='pet')
axs[0].set_title('Original Word Vectors')
axs[0].set_xlabel('Dimension')
axs[0].set_ylabel('Value')
axs[0].legend()
axs[0].grid(True)

# Plot the averaged vector
axs[1].plot([1, 2, 3], sentence_vector, marker='o', color='black', label='sentence')
axs[1].set_title('Averaged Sentence Vector')
axs[1].set_xlabel('Dimension')
axs[1].set_ylabel('Value')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.savefig('word2vec_visual_avg.png')
plt.show()
