import torch
import torch.nn as nn
import torch.nn.functional as F
import pytomography
from .U_MAPEM import Net



class U_BSREM(nn.Module):
    def __init__(self, beta, n_iter, scatter = None) -> None:
        super(U_BSREM, self).__init__()
        self.Asum = None
        self.n_iter = n_iter
        self.scatter = scatter
        self.beta = beta
        self.alpha_num= 1
        self.alpha_denom = 1
        self.net_liste = []
        self.net_liste = nn.ModuleList([Net() for _ in range(self.n_iter)])

    def set_Asum(self, Spect_sys, shape):
        self.Asum = Spect_sys.backward(torch.ones(shape).float().cuda())


    def forward(self, x, y, SPECT_sys, n_iter = None,callback = False):
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
        
           
            ybar = SPECT_sys.forward(out)
            alpha = 1/(1+ (self.alpha_num/self.alpha_denom)*iter) 
            precond = out/(self.Asum + pytomography.delta)
            if self.scatter is None:
                yratio = torch.div(y, ybar + pytomography.delta)
            else:
                yratio = torch.div(y, ybar + self.scatter + pytomography.delta)         
            grad_pnl = SPECT_sys.backward(yratio) - self.Asum         

            out = out + alpha * precond * (grad_pnl - u_rescaled)
            out = F.relu(out, inplace=True)
            if callback:
                obj_liste.append(out.clone().cpu())
        if callback:
            return obj_liste
        else:
            return out
    

class U_BSREM_JFB(nn.Module):
    def __init__(self, beta, n_iter, scatter = None, ct_use = None, grad_penalty = False) -> None:
        super(U_BSREM_JFB, self).__init__()
        self.Asum = None 
        self.n_iter = n_iter
        self.scatter = scatter
        self.beta = beta
        self.ct_use = ct_use
        grad_penalty = grad_penalty
        self.alpha_num= 1
        self.alpha_denom = 1
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
                
                    ybar = SPECT_sys.forward(out)
                    alpha = 1/(1+ (self.alpha_num/self.alpha_denom)*iter)
                    precond = out/(self.Asum + pytomography.delta)
                    if self.scatter is None:
                        yratio = torch.div(y, ybar + pytomography.delta)
                    else:
                        yratio = torch.div(y, ybar + self.scatter + pytomography.delta)
                    grad_pnl = SPECT_sys.backward(yratio) - self.Asum         
            

                    out = out + alpha * precond * (grad_pnl - u_rescaled)
                    out = F.relu(out, inplace=True)  
            else:
                
                mean = out.mean()
                std = out.std()
                out_scaled = (out - mean) / std if std != 0 else out
                u = self.net(out_scaled.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
                u_rescaled = u * std + mean
            
                ybar = SPECT_sys.forward(out)

                alpha = 1/(1+ (self.alpha_num/self.alpha_denom)*iter)
                precond = out/(self.Asum + pytomography.delta)
                if self.scatter is None:
                    yratio = torch.div(y, ybar + pytomography.delta)
                else:
                    yratio = torch.div(y, ybar + self.scatter + pytomography.delta)    
                grad_pnl = SPECT_sys.backward(yratio) - self.Asum           

                out = out + alpha * precond * (grad_pnl - u_rescaled)
                out = F.relu(out, inplace=True)
            if callback:
                obj_liste.append(out.clone().cpu())
        if callback:
            return obj_liste
        else:
            return out