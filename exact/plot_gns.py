import csv
import matplotlib.pyplot as plt

# Initialize lists to store the data
steps = []
ddp_gns_values = []
val_losses = []
lr_values = []

widths = [256, 512, 1024, 2048]
n_layer = 12
lrs =  [0.015625, 0.0078125, 0.00390625, 0.001953125]
#plt.figure(1)
for width in widths:
    plt.figure()
    for lr in lrs:
        # Read the CSV file
        try:
            with open(f'/p/scratch/cslfse/aach1/mup_logs/20k/layers_{n_layer}_width_{width}_lr_{lr}/log.csv', 'r') as file:
                reader = list(csv.DictReader(file))
                for row in reader[:-1]:
                    steps.append(int(row['step']))
                    ddp_gns_values.append(float(row['ddp/gns']))

                # Plot the data
                plt.plot(steps, ddp_gns_values, label=f'lr={lr}')
                print(max(ddp_gns_values))
                steps = []
                ddp_gns_values = []
        except: 
            #print(f'layers_{n_layer}_width_{width}_lr_{lr} does not exist')
            with open(f'/p/scratch/cslfse/aach1/mup_logs/20k/layers_{n_layer}_width_{width}_lr_{lr}/log_data.csv.tmp', 'r') as file:
                reader = list(csv.DictReader(file))
                for row in reader[:-1]:
                    steps.append(int(row['step']))
                    ddp_gns_values.append(float(row['ddp/gns']))

                # Plot the data
                plt.plot(steps, ddp_gns_values, label=f'lr={lr}')
                print(max(ddp_gns_values))
                steps = []
                ddp_gns_values = []

    # # Plot the data
    plt.xlabel('Step')
    plt.ylabel('ddp/gns')
    plt.ylim(0,200)
    plt.title('Step vs ddp/gns')
    plt.legend()
    plt.savefig(f'gns_plot_width_{width}.png', dpi = 300)

#plt.figure(2)
for width in widths:
    plt.figure()
    for lr in lrs:
        # Read the CSV file
        try:
            with open(f'/p/scratch/cslfse/aach1/mup_logs/20k/layers_{n_layer}_width_{width}_lr_{lr}/log.csv', 'r') as file:
                reader = list(csv.DictReader(file))
                for row in reader[:-1]:
                    steps.append(int(row['step']))
                    val_losses.append(float(row['train/lossf']))

                # Plot the data
                plt.plot(steps, val_losses, label=f'lr={lr}')
                steps = []
                val_losses = []
        except: 
            with open(f'/p/scratch/cslfse/aach1/mup_logs/20k/layers_{n_layer}_width_{width}_lr_{lr}/log_data.csv.tmp', 'r') as file:
                reader = list(csv.DictReader(file))
                for row in reader[:-1]:
                    steps.append(int(row['step']))
                    val_losses.append(float(row['train/lossf']))

                # Plot the data
                plt.plot(steps, val_losses, label=f'lr={lr}')
                steps = []
                val_losses = []
            #print(f'layers_{n_layer}_width_{width}_lr_{lr} does not exist')
    # # Plot the data
    plt.xlabel('Step')
    plt.ylabel('train/loss')
    plt.title('Step vs train/loss')
    plt.legend()
    plt.savefig(f'train_loss_width_{width}.png', dpi = 300)

# Add new figure for learning rates
# plt.figure(3)
for width in widths:
    plt.figure()
    for lr in lrs:
        # Read the CSV file
        try: 
            with open(f'/p/scratch/cslfse/aach1/mup_logs/20k/layers_{n_layer}_width_{width}_lr_{lr}/log.csv', 'r') as file:
                reader = list(csv.DictReader(file))
                for row in reader[:-1]:
                    steps.append(int(row['step']))
                    lr_values.append(float(row['lr']))  # Assuming 'lr' is the column name in your CSV

                # Plot the data
                plt.plot(steps, lr_values, label=f'lr={lr}')
                steps = []
                lr_values = []
        except:
            print(f'layers_{n_layer}_width_{width}_lr_{lr} does not exist')

    # Plot the learning rate data
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.title('Step vs Learning Rate')
    plt.legend()
    plt.savefig(f'learning_rate_width_{width}.png', dpi = 300)