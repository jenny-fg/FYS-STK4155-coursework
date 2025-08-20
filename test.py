#
#//  test.py
#//
#//
#//  Created by Jenny Guldvog on 20/08/2025.
#//

import matplotlib.pyplot as plt

# Eksempeldata
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# Lag et enkelt linjediagram
plt.plot(x, y)

# Legg til tittel og akse-labels
plt.title("Basic Matplotlib Plot")
plt.xlabel("X-akse")
plt.ylabel("Y-akse")

# Vis plottet
plt.show()


