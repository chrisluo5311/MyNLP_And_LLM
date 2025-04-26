import torch
import matplotlib.pyplot as plt

# 1. Create fake word vectors (each 3D)
cat_vec = torch.tensor([0.2, 0.1, 0.4])
dog_vec = torch.tensor([0.3, 0.0, 0.5])
pet_vec = torch.tensor([0.25, 0.05, 0.45])

# Stack into a tensor
word_vectors = torch.stack([cat_vec, dog_vec, pet_vec])  # (3, 3)

print("Original word vectors:")
print(word_vectors)

# 2. Average over words
avg_vector = word_vectors.mean(dim=0)

# 3. Max over words
max_vector = word_vectors.max(dim=0)[0]  # .max returns (values, indices), so take [0]

# 4. Combine (concatenate)
combined_vector = torch.cat([avg_vector, max_vector])  # Now size 6

print("\nAveraged vector:")
print(avg_vector)

print("\nMax vector:")
print(max_vector)

print("\nCombined vector (avg + max):")
print(combined_vector)

# 5. Plot (2x2 grid)
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Plot each original word vector
axs[0, 0].plot([1, 2, 3], cat_vec, marker='o', label='cat')
axs[0, 0].plot([1, 2, 3], dog_vec, marker='o', label='dog')
axs[0, 0].plot([1, 2, 3], pet_vec, marker='o', label='pet')
axs[0, 0].set_title('Original Word Vectors')
axs[0, 0].set_xlabel('Dimension')
axs[0, 0].set_ylabel('Value')
axs[0, 0].legend()
axs[0, 0].grid(True)

# Plot average vector
axs[0, 1].plot([1, 2, 3], avg_vector, marker='o', color='blue', label='average')
axs[0, 1].set_title('Averaged Sentence Vector')
axs[0, 1].set_xlabel('Dimension')
axs[0, 1].set_ylabel('Value')
axs[0, 1].legend()
axs[0, 1].grid(True)

# Plot max vector
axs[1, 0].plot([1, 2, 3], max_vector, marker='o', color='green', label='max')
axs[1, 0].set_title('Max Sentence Vector')
axs[1, 0].set_xlabel('Dimension')
axs[1, 0].set_ylabel('Value')
axs[1, 0].legend()
axs[1, 0].grid(True)

# Plot combined vector (6D)
axs[1, 1].plot(range(1, 7), combined_vector, marker='o', color='purple', label='avg+max')
axs[1, 1].set_title('Combined Sentence Vector (avg + max)')
axs[1, 1].set_xlabel('Dimension')
axs[1, 1].set_ylabel('Value')
axs[1, 1].legend()
axs[1, 1].grid(True)

plt.tight_layout()
plt.savefig('word2vec_visual_avg_max.png')
plt.show()
