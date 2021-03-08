import durations
import matplotlib.pyplot as plt

x = [x/1000 if x/1000 < 60 else 60 for x in durations.l]
print(max(x))
print(len(x))
n,bins,patches = plt.hist(x, bins=60)  # density=False would make counts
print(n,bins,patches)
plt.ylabel('duration')
plt.xlabel('Data')
plt.savefig('hist.png')
