import logging
from pathlib import Path
import sys
import torch
import AttnRes
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 1000
LR = 5e-4
IMAGEX = 227
IMAGEY = 235

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

if __name__ == "__main__":
    block_size = 45
    # Dataloader will provide batches of 8 tokens
    dataloader = AttnRes.dataloader.dataloader(
         data_dir = Path("data"), 
         block_size = block_size, 
         batch_size = 32
    )
    
    model_kwargs = dict(
         num_embeddings=50257,
         embedding_dim=2048,
         n_heads = 8,
         attn_layers = 5,
         dropout = 0.05
    )
    model = AttnRes.GPT.LanguageModel(**model_kwargs).to(DEVICE)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logging.info(f"Created a Language model with {param_count:,} parameters.")

    optim = torch.optim.AdamW(
         model.parameters(),
         lr = LR
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, 
        T_max=EPOCHS,  # Total number of iterations to reach minimum LR
        eta_min=LR*1e-4       # Minimum learning rate target
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
        for _ in tqdm(range(100)):
            # Get the training batch
            data = dataloader.get_batch("train")

            for i in range(block_size-2):
                
                inputs  = data[0][...,:i+2]
                targets = data[1][..., i+2].to(torch.long)

                optim.zero_grad()

                guess = model(inputs)

                error = loss(guess, targets)

                error.backward()
                optim.step()

                train_loss.append(error.detach().item())
        scheduler.step()
        metrics["train"].append( np.mean(train_loss) )

        model.eval()
        val_loss = []
        with torch.no_grad():
            for _ in tqdm(range(10)):
                # Get the training batch
                data = dataloader.get_batch("val")
    
                for i in range(block_size-2):
                    inputs  = data[0][...,:i+2]
                    targets = data[1][..., i+2].to(torch.long)
                    guess = model(inputs)
                    error = loss(guess, targets)
                    val_loss.append(error.detach().item())

        
        metrics["validation"].append( np.mean(val_loss) )

        logging.info(f"Error at the end of epoch: {metrics['train'][-1]} [Training]; {metrics['validation'][-1]} [Validation]")

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_kwargs": model_kwargs,
                "block_size": block_size,
                "epoch": e,
            },
            "checkpoint.pt",
        )
        plt.figure()
        plt.plot(metrics["train"], label="Training")
        plt.plot(metrics["validation"], label="Validation")
        plt.grid()
        plt.legend()
        plt.xlabel("Training Epoch [-]")
        plt.ylabel("Cross Entropy Loss [-]")
        plt.title("My GPT")
        plt.savefig("GPT_Training.png")
        plt.close()


    
    plt.show()


