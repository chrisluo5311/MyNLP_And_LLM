# Make sure you have matplotlib installed
# pip install matplotlib

import matplotlib.pyplot as plt

# Create a new figure and axis
fig, ax = plt.subplots()

# Set limits for better text placement visualization
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

# Add some text at a specific position (x=2, y=5)
ax.text(2, 5, 'Hello, World!', fontsize=14, color='blue')

# Add more text with customization
ax.text(6, 8, 'Matplotlib Text Example', fontsize=12, color='green', rotation=45)

# Title for context
ax.set_title('Simple Text Plotting Example')

# Show grid for better reference
ax.grid(True)

# Display the plot
plt.show()
