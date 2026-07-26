import logging
from pathlib import Path
import sys
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import AttnRes
import numpy as np
from tqdm import tqdm
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 25
IMAGEX = 227
IMAGEY = 235

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def load_data( 
            TrainDir= "../deep-learning/Datasets/MRI/Training", 
            ValidationDir= "../deep-learning/Datasets/MRI/Testing", 
            BatchSize = 32, 
            NumWorkers=2, 
            ImageDimX=IMAGEX, 
            ImageDimY=IMAGEY
        ) -> None:
        logging.info(f"Initializing Data Loaders.")
        trnsfrm = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            torchvision.transforms.RandomHorizontalFlip(torch.tensor([0.5])),
            transforms.Resize((ImageDimX, ImageDimY)),
            transforms.ToTensor()
        ])
        val_trnsfrm = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((ImageDimX, ImageDimY)),
            transforms.ToTensor()
        ])

        training_dataset = datasets.ImageFolder(root=TrainDir, transform=trnsfrm)
        validation_dataset = datasets.ImageFolder(root=ValidationDir, transform=val_trnsfrm)
        training_dataset = training_dataset
        validation_dataset = validation_dataset
        idx_identifier = training_dataset.class_to_idx

        logging.info(class_mapping := idx_identifier)

        training_data_loader = DataLoader(
            training_dataset,
            batch_size=BatchSize,
            shuffle=True,
            num_workers=NumWorkers,
            pin_memory=True
        )
        validation_data_loader = DataLoader(
            validation_dataset,
            batch_size=BatchSize,
            shuffle=True,
            num_workers=NumWorkers,
            pin_memory=True
        )

        return training_data_loader, validation_data_loader, idx_identifier

if __name__ == "__main__":
    train_loader, val_loader, idx_ident = load_data()

    model = AttnRes.ConvStem.ConvSparseAttentionResNet(
         in_h = IMAGEX,
         in_w = IMAGEY,
         outdim = len(idx_ident.keys()),
         resblocks = 4,
         resdim = 128,
         reslayers = 2,
         latentdim = 128
    ).to(DEVICE)

    optim = torch.optim.AdamW(
         model.parameters(),
         lr = 5e-3
    )

    metrics = {
         "train": [],
         "validation": []
    }
    loss = torch.nn.CrossEntropyLoss()

    for e in range(EPOCHS):
        train_loss = []
        logging.info(f"Starting epoch {e}")
        model.train()
        for inputs, targets in tqdm(train_loader):
            inputs  = (inputs + 0.05 * torch.rand_like(inputs)).to(DEVICE)
            targets = targets.to(DEVICE)

            optim.zero_grad()

            guess = model(inputs)

            error = loss(guess, targets)

            error.backward()
            optim.step()

            train_loss.append(error.detach().item())

        metrics["train"].append( np.mean(train_loss) )
        logging.info(f"Training Error:   {metrics["train"][-1]}")
        validation_loss = []
        model.eval()
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs  = inputs.to(DEVICE)
                targets = targets.to(DEVICE)
    
                guess = model(inputs)
    
                error = loss(guess, targets)
    
                validation_loss.append(error.detach().item())
        metrics["validation"].append( np.mean(validation_loss) )
        logging.info(f"Validation Error: {metrics["validation"][-1]}")

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(metrics["train"])
    plt.plot(metrics["validation"])
    plt.show()


