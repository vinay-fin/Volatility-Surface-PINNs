class SinAct(nn.Module):
    def __init__(self):
            super(SinAct, self).__init__() 

    def forward(self, x):
        return torch.sin(x)

class SFPINN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, sigma, activation):
        super(SFPINN, self).__init__()
        self.sigma = nn.Parameter(torch.tensor(sigma))
        self.first_linear = nn.Linear(in_dim, hidden_dim, bias=True)
        nn.init.normal_(self.first_linear.weight, mean=0.0, std=self.sigma.item())
        nn.init.constant_(self.first_linear.bias, 0.0)

        act_map = {"tanh": nn.Tanh(), "sigmoid": nn.Sigmoid(), "sin": SinAct()}
        self.act = act_map[activation]

        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers-2):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.output_layer = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        z = self.first_linear(x)
        src = torch.sin(2 * np.pi * z)
        for layer in self.hidden_layers:
            src = self.act(layer(src))
        return self.output_layer(src)
