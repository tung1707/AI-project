import timm
import torch.nn as nn
class Net(nn.Module):
    def __init__(self,pretrained=False):
        super(Net,self).__init__()
        self.output_type = ['infer','loss']
        arch = 'resnet18'

        dim = {'resnet18':512}
        dim = dim.get(arch,1280)
        self.model = timm.create_model(model_name=arch,pretrained=pretrained,in_chans = 3,num_classes=0 , global_pool='')
        self.target = nn.Linear(dim,1)

        self.dropout = nn.ModuleList([
            nn.Dropout(0.5) for i in range(5)
        ])
    
    def forward(self,batch):
        image = batch['image']
        batch_size = len(image)

        image = image.float()/255
        x =   self.model(image)
        pool = F.adaptive_avg_pool2d(x,1).reshape(batch_size,-1)

        if self.training:
            logit = 0
            for i in range(len(self.dropout)):
                logit += self.target(self.dropout[i](pool))
            logit = logit/len(self.dropout)
        else:
            logit = self.target(pool)
        
        # -------------------
        output = {}
        if 'loss' in self.output_type:
            target = batch['target']
            output['bce_loss'] = F.binary_cross_entropy_with_logits(logit.float(),target.float())
        
        if 'infer' in self.output_type:
            output['target'] = torch.sigmoid(logit)
        
        return output
    
def run_check_net():
        image_size = 256
        batch_size = 32

        batch = {
            'image':torch.from_numpy(np.random.uniform(-1,1,(batch_size,3,image_size,image_size))).float(),
            'target':torch.from_numpy(np.random.choice(2,(batch_size,1))).float()
        }

        net = Net(pretrained=False)

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True):
                output = net(batch)
        
        print('batch')
        for k,v in batch.items():
            print(f'{k:>32} : {v.shape}')
        print('output')

        for k,v in output.items():
            if 'loss' not in k:
                print(f'{k:>32} : {v.shape}')
        print('loss')
        for k,v in output.items():
            if 'loss' in k:
                print(f'{k:>32} : {v.item()}')

run_check_net()

