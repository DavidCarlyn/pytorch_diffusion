from argparse import ArgumentParser

import torch
import torch.nn as nn
from torchvision.transforms import ToTensor, ToPILImage, Resize

from PIL import Image

class DDPM(nn.Module):
    def __init__(self, beta_start, beta_end, T) -> None:
        super().__init__()
        self.betas = torch.linspace(0.0001, 0.02, args.T)
        self.alphas = 1 - self.betas

        self.recon_layers = []
        for _ in range(T):
            self.recon_layers.append(nn.Linear(4096, 4096))

    def diffuse(self, x_0, t):
        a_comp = self.alphas[:t].prod()

        self.epsilon = torch.normal(0, 1, size=x_0.shape)

        x_t = torch.sqrt(a_comp) * x_0 + (1-a_comp) * self.epsilon

        return x_t

    def denoise(self, x_t, t):
        return self.recon_layers[t](x_t)

    def forward(self, x_0, t):
        x_t = self.diffuse(x_0, t)
        x_0_prime = self.denoise(x_t, t)
        return x_0_prime

def save_img(x_t, path):
    out = torch.clamp(x_t, 0, 1)
    out = ToPILImage()(out)
    out.save(path)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--t", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()


    x = Image.open("dog.bmp")
    resize = Resize((64, 64))
    x = ToTensor()(resize(x)).unsqueeze(0).cuda()

    save_img(x[0], "org.png")

    loss_fn = nn.MSELoss()
    ddpm = DDPM(0.0001, 0.02, args.T).cuda()
    optimizer = torch.optim.Adam(ddpm.parameters(), lr=args.lr)
    for epoch in range(1000):
        t = torch.randint(0, args.T)
        out = ddpm(x, t)
        loss = loss_fn(ddpm.epsilon, out)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch: {epoch+1} | Loss: {loss.item()}")



