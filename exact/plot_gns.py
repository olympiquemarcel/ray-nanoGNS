import csv
import matplotlib.pyplot as plt

# Initialize lists to store the data
steps = []
ddp_gns_values = []
val_losses = []

widths = [256, 512, 1024]

plt.figure(1)
for width in widths:
    # Read the CSV file
    with open(f'out/width_{width}_lr_0.00390625/log.csv', 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            steps.append(int(row['step']))
            ddp_gns_values.append(float(row['ddp/gns']))

    # Plot the data
    plt.plot(steps, ddp_gns_values, label=f'width={width}')
    steps = []
    ddp_gns_values = []

# # Plot the data
plt.xlabel('Step')
plt.ylabel('ddp/gns')
plt.title('Step vs ddp/gns')
plt.legend()
plt.savefig("gns_plot_widths.png", dpi = 300)

plt.figure(2)
for width in widths:
    # Read the CSV file
    with open(f'out/width_{width}_lr_0.00390625/log.csv', 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            steps.append(int(row['step']))
            val_losses.append(float(row['train/lossf']))

    # Plot the data
    plt.plot(steps, val_losses, label=f'width={width}')
    steps = []
    val_losses = []

# # Plot the data
plt.xlabel('Step')
plt.ylabel('train/loss')
plt.title('Step vs val/loss')
plt.legend()
plt.savefig("train_loss_widths.png", dpi = 300)

