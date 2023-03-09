from argparse import ArgumentParser

import torch
import torch.nn as nn
from torchvision.transforms import ToTensor, ToPILImage, Resize

from PIL import Image

import matplotlib.pyplot as plt

def embed_timestep(x, t):
    ch = x.shape[1]
    ch_h = ch // 2
    if ch_h <= 1:
        emb = torch.log(torch.tensor(10000)).cuda()
    else:
        emb = torch.log(torch.tensor(10000)).cuda()/(ch_h-1)
    emb = torch.exp(torch.arange(0, ch_h).cuda()*-emb)
    emb = torch.tensor(t).unsqueeze(0).cuda() * emb.unsqueeze(0)
    emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=1)
    
    if ch % 2 == 1:
        emb = torch.cat((emb, torch.zeros_like(emb)[:, :1]), dim=1)
    
    emb = emb.unsqueeze(2).unsqueeze(3).repeat((1, 1, x.shape[2], x.shape[3]))
    return x + emb

class DDPM(nn.Module):
    def __init__(self, beta_start, beta_end, T) -> None:
        super().__init__()
        self.betas = torch.linspace(beta_start, beta_end, args.T).cuda()
        self.alphas = 1 - self.betas

        self.encode = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, padding=0),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, padding=0),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
        )

        self.decode = nn.Sequential(
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 2, 2, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, 2, 2, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 3, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(3, 3, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(3, 3, 3, 1, 1),
            nn.Tanh()
        )

        self.linear_encode = nn.Conv2d(3, 3, 1, 1, 0)
        self.linear_decode = nn.Conv2d(64, 64, 1, 1, 0)
        
    def diffuse(self, x_0, t):
        a_comp = self.alphas[:t+1].prod()

        self.epsilon = torch.normal(0, 1, size=x_0.shape).cuda()

        x_t = torch.sqrt(a_comp) * x_0 + (1-a_comp) * self.epsilon

        return x_t

    def denoise(self, x_t, t):
        z = self.encode(embed_timestep(x_t, t))
        out = self.decode(embed_timestep(z, t)) 
        #z = self.encode(self.linear_encode(embed_timestep(x_t, t)))
        #out = self.decode(self.linear_decode(embed_timestep(z, t))) 
        #out = (out + 1)/2
        return out

    def forward(self, x_0, t):
        x_t = self.diffuse(x_0, t)
        x_0_prime = self.denoise(x_t, t)
        return x_0_prime

def save_img(x_t, path):
    out = torch.clamp(x_t, 0, 1)
    out = ToPILImage()(out)
    out.save(path)

def plot(losses):
    plt.plot(losses)
    plt.savefig("loss_plot.png")
    plt.close()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.0001)
    args = parser.parse_args()


    x = Image.open("dog.bmp")
    resize = Resize((64, 64))
    x = ToTensor()(resize(x)).unsqueeze(0).cuda()

    save_img(x[0], "org.png")

    loss_fn = nn.MSELoss()
    ddpm = DDPM(0.0001, 0.02, args.T).cuda()
    optimizer = torch.optim.Adam(ddpm.parameters(), lr=args.lr)
    losses = []
    for epoch in range(args.epochs):
        total_loss = 0
        for k in range(args.T):
            t = torch.randint(0, args.T, size=(1,)).cuda()
            out = ddpm(x, t)
            loss = loss_fn(ddpm.epsilon, out)# * (t+1)/args.T

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        print(f"Epoch: {epoch+1} | Loss: {total_loss}")
        losses.append(total_loss)
        plot(losses)

        with torch.no_grad():
            x_T = torch.normal(0, 1, size=x.shape).cuda()
            for t in range(args.T-1, -1, -1):
                alpha = ddpm.alphas[t]
                noise_pred = ddpm.denoise(x_T, t)
                z = torch.normal(0, 1, size=x_T.shape).cuda()
                if t == 0:
                    z = torch.zeros_like(x_T)
                x_T = (1/torch.sqrt(alpha)) * (x_T - (1 - alpha)/torch.sqrt(1-ddpm.alphas[:t+1].prod()) * noise_pred) + torch.sqrt(ddpm.betas[t])*z
            save_img((x_T[0]+1)/2, "recon.png")
            torch.save(ddpm.state_dict(), "model.pt")




