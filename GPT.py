import logging
from pathlib import Path
import sys
import torch
import AttnRes
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_TOKENS = 3.5e9 # Train this model an ~3.5 Billion tokens
LR = 5e-4
LOG_EVERY = 100

tqdm.format_num = lambda n: f"{int(n):,}" if isinstance(n, (int, float)) else str(n)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

if __name__ == "__main__":
    block_size = 30
    # Dataloader will provide batches of 8 tokens
    dataloader = AttnRes.dataloader.dataloader(
         data_dir = Path("data"), 
         block_size = block_size, 
         batch_size = 32
    )
    
    model_kwargs = dict(
         num_embeddings=50257,
         embedding_dim=1024,
         n_heads = 4,
         max_seq_len = block_size,
         attn_layers = 3,
         dropout = 0.05
    )
    
    model = AttnRes.GPT.LanguageModel(**model_kwargs).to(DEVICE)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logging.info(f"Created a Language model with {param_count:,} parameters.")

    optim = torch.optim.AdamW(
         model.parameters(),
         lr = LR
    )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optim, 
    #     T_max=EPOCHS,  # Total number of iterations to reach minimum LR
    #     eta_min=LR*1e-4       # Minimum learning rate target
    # )
    metrics = {
         "train": [],
         "validation": []
    }
    chunk = 100
    prev_token_count = 0
    num_tokens = []
    loss = torch.nn.CrossEntropyLoss()
    best_val_loss = 1000
    
    train_loss = []
    logging.info(f"Starting training on {MAX_TOKENS:,} tokens")

    with tqdm(total=MAX_TOKENS) as pbar:
        while prev_token_count < MAX_TOKENS:
            model.train()
            # Get the training batch
            inputs, targets = dataloader.get_batch("train")
            B, T = inputs.shape

            inputs  = inputs
            targets = targets.to(torch.long)

            optim.zero_grad()

            guess = model(inputs)
            error = loss(guess.view(B*T, -1), targets.view(B*T))

            error.backward()
            optim.step()

            train_loss.append(error.detach().item())
            pbar.update(B*T)
            # scheduler.step()
            
            prev_token_count = prev_token_count + B*T
            

            # Log off plotting points every 100 steps
            if len(train_loss) >= LOG_EVERY:
                metrics["train"].append( np.mean(train_loss) )
                train_loss = []

                model.eval()
                with torch.no_grad():
                    # Get the training batch
                    data = dataloader.get_batch("val")
        
                    inputs  = data[0]
                    targets = data[1].to(torch.long)
                    guess = model(inputs)
                    
                    error = loss(guess.view(B*T, -1), targets.view(B*T))
    
                    metrics["validation"].append( error.detach().item() )
  
                num_tokens.append(prev_token_count)

                if metrics["validation"][-1] < best_val_loss:
                    best_val_loss = metrics["validation"][-1]
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "model_kwargs": model_kwargs,
                            "block_size": block_size,
                            "num_tokens": prev_token_count,
                        },
                        "checkpoint.pt",
                    )
                    logging.debug(f"Saved model checkpoint with a validation loss of {metrics["validation"][-1]}")

                plt.figure()
                plt.plot(np.array( num_tokens ) / 1e6, metrics["train"], label="Training")
                plt.plot(np.array( num_tokens ) / 1e6, metrics["validation"], label="Validation")
                plt.grid()
                plt.legend()
                plt.xlabel("Millions of Tokens")
                plt.ylabel("Cross Entropy Loss [-]")

                plt.ylim([0,10])
                plt.title("GPT Training History")
                plt.savefig("GPT_Training.png")
                plt.close("all")


    
    plt.show()


