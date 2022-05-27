import copy
import torch.nn as nn

class EMAHelper(object):
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(inner_module.config).to(inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        # module_copy = copy.deepcopy(module)
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


# import glob, torch, tqdm
# ckpt_files = sorted(glob.glob("*.pt"))
# for file in tqdm.tqdm(ckpt_files):
#     a = torch.load(file)
#     a[0]['module.unet.all_modules.52.Norm_0.weight'] = a[0].pop('module.unet.all_modules.52.weight')
#     a[0]['module.unet.all_modules.52.Norm_0.bias'] = a[0].pop('module.unet.all_modules.52.bias')
#     a[-1]['unet.all_modules.52.Norm_0.weight'] = a[-1].pop('unet.all_modules.52.weight')
#     a[-1]['unet.all_modules.52.Norm_0.bias'] = a[-1].pop('unet.all_modules.52.bias')
#     torch.save(a, file)
