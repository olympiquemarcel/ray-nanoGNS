import csv
import matplotlib.pyplot as plt

# Initialize lists to store the data
steps = []
ddp_gns_values = []

# Read the CSV file
with open('out/test/log.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        steps.append(int(row['step']))
        ddp_gns_values.append(float(row['ddp/gns']))

# Plot the data
plt.plot(steps, ddp_gns_values, label='ddp/gns')
plt.xlabel('Step')
plt.ylabel('ddp/gns')
plt.title('Step vs ddp/gns')
plt.legend()
plt.savefig("gns_plot.png", dpi = 300)
plt.show()
