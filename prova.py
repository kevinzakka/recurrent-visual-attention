import numpy as np
import matplotlib.pyplot as plt

sizes = ['2x2', '4x4', '6x6', '8x8']
accuracies = [0.9, 0.85, 0.8, 0.75]

x = np.arange(len(sizes))
plt.plot(x, accuracies, marker='o')
plt.xticks(x, sizes)
plt.xlabel('Size')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Size')
plt.show()