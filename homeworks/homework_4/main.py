import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import random_split
import os
from prettytable import PrettyTable
from torchvision.datasets import FashionMNIST

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, resize_transform=None, target_transform=None):
        self.data = data
        self.resize_transform = resize_transform
        self.target_transform = target_transform
        self.targets = [label for _, label in data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]

        if self.target_transform:
            original_image = self.target_transform(image)

        if self.resize_transform:
            resized_image = self.resize_transform(image)

        return resized_image, original_image,label


# resize_transform = transforms.Compose([
#     transforms.Resize((7,7)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.1307,), (0.3081,)),
# ])

# target_transform_train = transforms.Compose([
#   transforms.ToTensor(),
#   transforms.Normalize((0.1307,), (0.3081,)),
# ])

# target_transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.1307,), (0.3081,)),
# ])


resize_transform = transforms.Compose([
    transforms.Resize((7,7)),
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5)),
])

target_transform_train = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize((0.5), (0.5)),
])

target_transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5)),
])


train_dataset_original = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True)
test_dataset_original  = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True)


train_dataset = CustomDataset(train_dataset_original,resize_transform=resize_transform, target_transform=target_transform_train)
test_dataset = CustomDataset(test_dataset_original, resize_transform=resize_transform, target_transform=target_transform_test)



train_size = int(0.8 * len(train_dataset))
validation_size = len(train_dataset) - train_size
train_dataset, validation_dataset = random_split(train_dataset, [train_size, validation_size])


BATCH_SIZE = 256
train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=BATCH_SIZE, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


class Encoder(nn.Module):
    def __init__(self, encoded_space_dim):
        super().__init__()
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=1, padding=1),  # 7x7 -> 7x7
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=1, padding=0),  # 7x7 -> 5x5
            nn.BatchNorm2d(16),
            nn.ReLU(True),
        )
        self.flatten = nn.Flatten(start_dim=1)

        self.encoder_lin = nn.Sequential(
            nn.Linear(16 * 5 * 5, 128),  # 16 --> last conv layer, 5--> 5x5 last conv dimension
            nn.ReLU(True),
            nn.Linear(128, encoded_space_dim)
        )

    def forward(self, x):
        #print(f"Shape initially inside encoder: {x.shape}")
        x = self.encoder_cnn(x)
        #print(f"Shape after encoder cnn: {x.shape}")
        x = self.flatten(x)
        #print(f"Shape after flatten: {x.shape}")
        x = self.encoder_lin(x)
        #print(f"Shape after encoder linear: {x.shape}")
        return x


class Decoder(nn.Module):
    def __init__(self, encoded_space_dim):
        super().__init__()
        #self.decoder_lin = nn.Linear(128, 16 * 5 * 5)

        self.decoder_lin = nn.Sequential(
           nn.Linear(encoded_space_dim, 128),
           nn.ReLU(True),
           nn.Linear(128, 16*5*5),
           nn.ReLU(True)
        )
        self.unflatten = nn.Unflatten(1, (16, 5, 5))

        # [5x5] to [7x7]
        self.upconv1 =nn.Sequential(
          nn.ConvTranspose2d(16,16, kernel_size=3, stride=1, padding=0),
          nn.BatchNorm2d(16),
          nn.ReLU(True)
        )
        self.upconv2 = nn.Sequential(
          nn.ConvTranspose2d(16,8,kernel_size=3, stride=1, padding=1),
          nn.BatchNorm2d(8),
          nn.ReLU(True)
        )
        self.upconv3 = nn.Sequential(
          nn.ConvTranspose2d(8, 4, kernel_size=4, stride=2, padding=1),
          nn.BatchNorm2d(4),
          nn.ReLU(True)
        )
        self.final_conv = nn.Sequential(
          nn.ConvTranspose2d(4, 1, kernel_size=4, stride=2, padding=1),
        #   nn.BatchNorm2d(1),
        #   nn.ReLU(True)
        )

    def forward(self, x):
        #print(f"Shape initially inside decoder: {x.shape}")
        x = self.decoder_lin(x)
        #print(f"Shape after decoder linear: {x.shape}")
        x = self.unflatten(x)
        #print(f"Shape after unflatten: {x.shape}")

        x = self.upconv1(x)
        #print(f"Shape after first convolutional: {x.shape}")
        x = self.upconv2(x)
        #print(f"Shape after second conv: {x.shape}")
        x = self.upconv3(x)
        #print(f"Shape after third conv: {x.shape}")
        x = self.final_conv(x)
        #print(f"Shape after final conv: {x.shape}")
        #x = torch.sigmoid(x)

        #print(f"Shape after sigmoid: {x.shape}")
        return x

class Encoder_2(nn.Module):
    def __init__(self, encoded_space_dim):
        super().__init__()
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
        )
        self.flatten = nn.Flatten(start_dim=1)
        self.encoder_lin = nn.Sequential(
            # nn.Linear(16 * 7 * 7, 128),
            # nn.ReLU(True),
            nn.Linear(32*5*5, encoded_space_dim),
            # nn.Linear(128, encoded_space_dim),
            nn.ReLU(True),
        )

    def forward(self, x):
        x = self.encoder_cnn(x)
        #print(f"Shape after encoder cnn: {x.shape}")
        x = self.flatten(x)
        #print(f"Shape after flatten: {x.shape}")
        x = self.encoder_lin(x)
        #print(f"Shape after encoder lin: {x.shape}")
        return x

class Decoder_2(nn.Module):
    def __init__(self, encoded_space_dim):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            # nn.Linear(encoded_space_dim, 128),
            nn.Linear(encoded_space_dim, 32*5*5),
            nn.ReLU(True),
            # nn.Linear(128, 16*7*7),
            # nn.ReLU(True),
        )
        self.unflatten = nn.Unflatten(1, (32, 5, 5))

        self.upconv1 =nn.Sequential(
          nn.ConvTranspose2d(32,16, kernel_size=3, stride=1, padding=0),
          nn.BatchNorm2d(16),
          nn.ReLU(True)
        )
        self.upconv2 = nn.Sequential(
          nn.ConvTranspose2d(16,8,kernel_size=3, stride=1, padding=1),
          nn.BatchNorm2d(8),
          nn.ReLU(True)
        )
        self.upconv3 = nn.Sequential(
          nn.ConvTranspose2d(8, 4, kernel_size=4, stride=2, padding=1),
          nn.BatchNorm2d(4),
          nn.ReLU(True)
        )
        self.final_conv = nn.Sequential(
          nn.ConvTranspose2d(4, 1, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        #print(f"Shape initially inside decoder: {x.shape}")
        x = self.decoder_lin(x)
        #print(f"Shape after decoder linear: {x.shape}")
        x = self.unflatten(x)
        #print(f"Shape after unflatten: {x.shape}")

        x = self.upconv1(x)
        #print(f"Shape after first convolutional: {x.shape}")
        x = self.upconv2(x)
        #print(f"Shape after second conv: {x.shape}")
        x = self.upconv3(x)
        #print(f"Shape after third conv: {x.shape}")
        x = self.final_conv(x)
        #print(f"Shape after final conv: {x.shape}")
        #x = torch.sigmoid(x)
        return x

class Encoder_3(nn.Module):
    def __init__(self, encoded_space_dim):
        super().__init__()
        # one convolutional layer
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # 7x7 -> 4x4
            nn.BatchNorm2d(16),
            nn.ReLU(True),
        )
        self.flatten = nn.Flatten(start_dim=1)
        self.encoder_lin = nn.Sequential(
            nn.Linear(16 * 4 * 4, 128),  # 16 channels and 4x4 dimension
            nn.ReLU(True),
            nn.Linear(128, encoded_space_dim)
        )

    def forward(self, x):
        x = self.encoder_cnn(x)
        #print(f"Shape after encoder cnn: {x.shape}")
        x = self.flatten(x)
        #print(f"Shape after flatten: {x.shape}")
        x = self.encoder_lin(x)
        #print(f"Shape after encoder lin: {x.shape}")
        return x


class Decoder_3(nn.Module):
    def __init__(self, encoded_space_dim):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 16*4*4),  # 4x4x16
            nn.ReLU(True)
        )
        self.unflatten = nn.Unflatten(1, (16, 4, 4))

        # [4x4] to [7x7]
        self.upconv1 = nn.Sequential(
            nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, padding=1),  # 4x4 -> 7x7
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )
        # [7x7] to [14x14]
        self.upconv2 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),  # 7x7 -> 14x14
            nn.BatchNorm2d(8),
            nn.ReLU(True)
        )
        # [14x14] to [28x28]
        self.final_conv = nn.Sequential(
            nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1),  # 14x14 -> 28x28
            # nn.BatchNorm2d(1),
            # nn.ReLU(True)
        )

    def forward(self, x):
        x = self.decoder_lin(x)
        #print(f"Shape after decoder lin: {x.shape}")
        x = self.unflatten(x)
        #print(f"Shape after unflatten: {x.shape}")
        x = self.upconv1(x)
        #print(f"Shape after conv1: {x.shape}")
        x = self.upconv2(x)
        #print(f"Shape after conv2: {x.shape}")
        x = self.final_conv(x)
        #print(f"Shape after conv3: {x.shape}")
        return x


device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f'Selected device: {device}')

def train_epoch(encoder, decoder, device, dataloader, loss_fn, optimizer):
    # Set train mode for both the encoder and the decoder
    encoder.train()
    decoder.train()
    train_loss = []
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for resized_image, original_image, _ in dataloader: # with "_" we just ignore the labels (the second element of the dataloader tuple)
        # Move tensor to the proper device

        resized_image = resized_image.to(device)
        #print(f"Shape of resized image: {resized_image.shape}")
        # Encode data
        encoded_data = encoder(resized_image)
        #print(f"Shape of encoded data: {encoded_data.shape}")
        # Decode data
        decoded_data = decoder(encoded_data)
        #print(f"Shape of decoded data: {decoded_data.shape}")
        # Evaluate loss
        #optimizer.zero_grad()
        original_image = original_image.to(device)
        loss = loss_fn(decoded_data, original_image)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        print('\t partial train loss (single batch): %f' % (loss.item()))
        train_loss.append(loss.detach().cpu().numpy())

    return encoder, decoder, np.mean(train_loss)



def test_epoch(encoder, decoder, device, dataloader, loss_fn):
    # Set evaluation mode for encoder and decoder
    encoder.eval()
    decoder.eval()
    with torch.no_grad(): # No need to track the gradients
        # Define the lists to store the outputs for each batch
        conc_out = []
        conc_label = []
        for resized_image,original_image, _ in dataloader:
            # Move tensor to the proper device
            resized_image = resized_image.to(device)
            # Encode data
            encoded_data = encoder(resized_image)
            # Decode data
            decoded_data = decoder(encoded_data)
            # Append the network output and the original image to the lists
            conc_out.append(decoded_data)
            original_image = original_image.to(device)
            conc_label.append(original_image)
        # Create a single tensor with all the values in the lists
        conc_out = torch.cat(conc_out)
        conc_label = torch.cat(conc_label)
        # Evaluate global loss
        val_loss = loss_fn(conc_out, conc_label)
        print(f"val loss: {val_loss}")
    return val_loss.data.detach().cpu().numpy()



def plot_ae_outputs_same_set(encoder, decoder, test_dataset, device, indices, model_name, epoch):

    model_dir = f'./reconstructed_images/{model_name}'
    os.makedirs(model_dir, exist_ok=True)

    plt.figure(figsize=(16, 4.5))

    for i, idx in enumerate(indices):
    
        resized_img, original_img, _ = test_dataset[idx]

    
        resized_img = resized_img.unsqueeze(0).to(device)

        
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            rec_img = decoder(encoder(resized_img))

        
        ax = plt.subplot(2, len(indices), i + 1)
        plt.imshow(original_img.squeeze().numpy(), cmap='gray')
        ax.axis('off')
        if i == len(indices) // 2:
            ax.set_title('Original images')

      
        ax = plt.subplot(2, len(indices), len(indices) + i + 1)
        plt.imshow(rec_img.cpu().squeeze().numpy(), cmap='gray')
        ax.axis('off')
        if i == len(indices) // 2:
            ax.set_title('Reconstructed images with same indices')

   
    filename = f'{model_dir}/{model_name}_epoch_{epoch}.png'
    plt.savefig(filename)
    #plt.show()

def plot_and_save_reconstructed_samples_for_epoch(encoder, decoder, test_dataset, device, model_name, epoch, num_classes=10):
    
    model_dir = f'./resized_predicted_original/{model_name}'
    os.makedirs(model_dir, exist_ok=True)
    
    plt.figure(figsize=(3 * num_classes, 9)) 

    for class_idx in range(num_classes):
     
        class_indices = [i for i, (_, _, label) in enumerate(test_dataset) if label == class_idx]
        sample_idx = class_indices[0] 


        resized_img, original_img, _ = test_dataset[sample_idx]

        resized_img = resized_img.unsqueeze(0).to(device)

 
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            rec_img = decoder(encoder(resized_img))

        # Plot 7x7 input image
        ax = plt.subplot(3, num_classes, class_idx + 1)
        plt.imshow(resized_img.cpu().squeeze().numpy(), cmap='gray')
        ax.axis('off')
        if class_idx == 4:
            ax.set_title('7x7 Input', fontsize=38)

        # Plot 28x28 reconstructed image
        ax = plt.subplot(3, num_classes, num_classes + class_idx + 1)
        plt.imshow(rec_img.cpu().squeeze().numpy(), cmap='gray')
        ax.axis('off')
        if class_idx == 4:
            ax.set_title('28x28 Predicted Output', fontsize=38)

        # Plot 28x28 desired (original) output
        ax = plt.subplot(3, num_classes, 2 * num_classes + class_idx + 1)
        plt.imshow(original_img.squeeze().numpy(), cmap='gray')
        ax.axis('off')
        if class_idx == 4:
            ax.set_title('28x28 Desired Output', fontsize=38)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  


    filename = f'{model_dir}/{model_name}_epoch_{epoch}_reconstructed_samples.png'
    plt.savefig(filename)
    plt.close() 
    #plt.show()



import matplotlib.pyplot as plt
import torch

def plot_losses(diz_loss, model_name):
    losses_dir = './losses'
    os.makedirs(losses_dir, exist_ok=True)

    train_losses = diz_loss[f'{model_name}_model_train']
    test_losses = diz_loss[f'{model_name}_model_test']
    epochs = range(1, len(train_losses) + 1)

    train_losses = [loss.cpu().numpy() if isinstance(loss, torch.Tensor) else loss for loss in train_losses]
    test_losses = [loss.cpu().numpy() if isinstance(loss, torch.Tensor) else loss for loss in test_losses]

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label='Training loss')
    plt.plot(epochs, test_losses, label='Test loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    filename = f'{losses_dir}/model_{model_name}_loss.png'
    plt.savefig(filename)

def report_model_info_as_table(model_pairs, diz_loss):
    report_dir="./report_table"
    os.makedirs(report_dir, exist_ok=True)

    # Create a table
    table = PrettyTable()
    table.field_names = ['Model', 'Total Parameter Count', 'Final Train MSE', 'Final Test MSE']

    for i, (encoder, decoder) in enumerate(model_pairs):
        total_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad) \
                       + sum(p.numel() for p in decoder.parameters() if p.requires_grad)
        train_loss = diz_loss[f"{i}_model_train"][-1]
        test_loss = diz_loss[f"{i}_model_test"][-1]
        
        # Add data to the table
        table.add_row([f'Model {i+1}', total_params, train_loss, test_loss])

    # Save the table as an image inside the report directory
    report_filename = os.path.join(report_dir, "model_performance_report.png")
    with open(report_filename, 'w') as f:
        f.write(str(table))


if __name__ == "__main__":

    num_classes = 10 
    indices_per_class = 1 
    fixed_indices = []
    for class_idx in range(num_classes):
        class_indices = np.where(np.array(test_dataset.targets) == class_idx)[0]
        class_specific_indices = np.random.choice(class_indices, indices_per_class, replace=False)
        fixed_indices.extend(class_specific_indices)

    encoder = Encoder(64).to(device)
    decoder = Decoder(64).to(device)

    encoder_2= Encoder_2(64).to(device)
    decoder_2 = Decoder_2(64).to(device)

    encoder_3= Encoder_3(64).to(device)
    decoder_3 = Decoder_3(64).to(device)

    loss_fn = nn.MSELoss()
  
    diz_loss = {}
    models = [(encoder, decoder), (encoder_2, decoder_2), (encoder_3, decoder_3)]

    for i, model in enumerate(models):
        enc, dec = model
        loss_fn = nn.MSELoss()
        diz_loss[f"{i}_model_train"] = []
        diz_loss[f"{i}_model_test"] = []
        optimizer = torch.optim.Adam(params=list(enc.parameters()) + list(dec.parameters()), lr=0.001)
        for epoch in range(15):
            enc, dec, train_loss = train_epoch(enc, dec, device=device,optimizer=optimizer, dataloader=train_loader, loss_fn=loss_fn) 
            diz_loss[f"{i}_model_train"].append(train_loss)
            test_loss = test_epoch(enc, dec, device, test_loader, loss_fn)
            diz_loss[f"{i}_model_test"].append(test_loss)
            #print('\n Model_{i} EPOCH {epoch}/{15} \t train loss {} \t val loss {}'.format(epoch + 1, epoch,train_loss,val_loss))
            plot_and_save_reconstructed_samples_for_epoch(enc,dec, test_dataset,device, f"model_{i}", epoch)

    
    report_model_info_as_table(model_pairs=models, diz_loss=diz_loss)
    #print(diz_loss)
    for i in range(len(models)):
        plot_losses(diz_loss=diz_loss, model_name=i)

