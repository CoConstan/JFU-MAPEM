import torch
import torch.nn as nn
import torch.nn.functional as F
import pytomography

class Net(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=16,
                               kernel_size=(3,3,3), stride=(1,1,1),
                               padding=(1,1,1))
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=32,
                               kernel_size=(3,3,3), stride=(1,1,1),
                               padding=(1,1,1))
        self.conv3 = nn.Conv3d(in_channels=32, out_channels=64,
                                 kernel_size=(3,3,3), stride=(1,1,1),
                                 padding=(1,1,1))
        self.conv4 = nn.Conv3d(in_channels=64, out_channels=32,
                               kernel_size=(3,3,3), stride=(1,1,1),
                               padding=(1,1,1))
        self.conv5 = nn.Conv3d(in_channels=32, out_channels=16,
                                 kernel_size=(3,3,3), stride=(1,1,1),
                                 padding=(1,1,1))
        self.conv6 = nn.Conv3d(in_channels=16, out_channels=1,
                               kernel_size=(3,3,3), stride=(1,1,1),
                               padding=(1,1,1))
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = F.relu(self.conv5(out))
        out = self.conv6(out)
        return out

class U_MAPEM(nn.Module):
    def __init__(self, beta, n_iter, scatter = None) -> None:
        super(U_MAPEM, self).__init__()
        self.Asum = None 
        self.n_iter = n_iter
        self.scatter = scatter
        self.beta = beta
        self.net_liste = nn.ModuleList([Net() for _ in range(self.n_iter)])
        
    def set_Asum(self, Spect_sys, shape):
        self.Asum = Spect_sys.backward(torch.ones(shape).float().cuda())

    def forward(self, x, y, SPECT_sys, n_iter = None, callback = False):
        if self.Asum is None:
            raise ValueError("Asum is not set")
        
        self.Asum[self.Asum == 0] = float('Inf')
        out = x
        n_iter = self.n_iter if n_iter is None else n_iter

        if callback:
            obj_liste=[]

        for iter in range(n_iter):
            net = self.net_liste[iter]
            
            mean = out.mean()
            std = out.std()
            out_scaled = (out - mean) / std if std != 0 else out
            u = net(out_scaled.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
            u_rescaled = u * std + mean
            with torch.no_grad():
                ybar = SPECT_sys.forward(out)
                if self.scatter is None:
                    yratio = torch.div(y, ybar + pytomography.delta)
                else:
                    yratio = torch.div(y, ybar + self.scatter + pytomography.delta)
            back = SPECT_sys.backward(yratio)
            reg = self.beta * (u_rescaled)
            out = torch.multiply(out, torch.div(back, self.Asum + reg))  #+ pytomography.delta))
            out = F.relu(out, inplace=True)
            if callback:
                obj_liste.append(out.clone().cpu())

        if callback:
            return obj_liste
        else:
            return out


class U_MAPEM_JFB(nn.Module):
    def __init__(self, beta, n_iter, scatter = None) -> None:
        super(U_MAPEM_JFB, self).__init__()
        self.Asum = None # at each epoch, this will be updated
        self.n_iter = n_iter
        self.scatter = scatter
        self.beta = beta
        self.net = Net()
        
    def set_Asum(self, Spect_sys, shape):
        self.Asum = Spect_sys.backward(torch.ones(shape).float().cuda())

    def forward(self, x, y, SPECT_sys, n_iter = None, callback = False):
        if self.Asum is None:
            raise ValueError("Asum is not set")
        self.Asum[self.Asum == 0] = float('Inf')
        out = x
        n_iter = self.n_iter if n_iter is None else n_iter
        if callback:
            obj_liste=[]
        for iter in range(n_iter):
            if self.training:
                with torch.set_grad_enabled(iter == n_iter-1):
                    mean = out.mean()
                    std = out.std()
                    out_scaled = (out - mean) / std if std != 0 else out
                    u = self.net(out_scaled.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
                    u_rescaled = u * std + mean
                    with torch.no_grad():
                        ybar = SPECT_sys.forward(out)
                        if self.scatter is None:
                            yratio = torch.div(y, ybar + pytomography.delta)
                        else:
                            yratio = torch.div(y, ybar + self.scatter + pytomography.delta)
                        back = SPECT_sys.backward(yratio)
                    reg = self.beta * (u_rescaled)
                    out = torch.multiply(out, torch.div(back, self.Asum + reg))
                    out = F.relu(out, inplace=True)
            else:
                mean = out.mean()
                std = out.std()
                out_scaled = (out - mean) / std if std != 0 else out
                u = self.net(out_scaled.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
                u_rescaled = u * std + mean
                with torch.no_grad():
                    ybar = SPECT_sys.forward(out)
                    if self.scatter is None:
                        yratio = torch.div(y, ybar + pytomography.delta)
                    else:
                        yratio = torch.div(y, ybar + self.scatter + pytomography.delta)
                    back = SPECT_sys.backward(yratio)
                reg = self.beta * (u_rescaled)
                out = torch.multiply(out, torch.div(back, self.Asum + reg))
                out = F.relu(out, inplace=True)
            
            if callback:
                obj_liste.append(out.clone().cpu())

        if callback:
            return obj_liste
        else:
            return out
