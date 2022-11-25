import torch
import torch.nn as nn




class Classfier(nn.Module):
    def __init__(self):
        super(Classfier,self).__init__()
        self.text_linear=nn.Linear(512,2)
        self.img_linear = nn.Linear(512,2)
        self.score=nn.Linear(1024,2)
        self.softmax=nn.Softmax(dim=1)
        self.dropout=nn.Dropout(p=0.5)


    def forward(self,text,img):
        output=torch.cat((text,img),dim=1)
        score=self.score(output).unsqueeze(1)
        text=self.text_linear(text).unsqueeze(1)
        img=self.img_linear(img).unsqueeze(1)
        output=torch.cat((text,img),dim=1)
        output=torch.bmm(score,output).squeeze(1)
        output = self.softmax(output)
        output=self.dropout(output)

        return output

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight.data)
                # m.weight.data.normal_(0,0.01)
                m.bias.data.zero_()

# if __name__ == '__main__':
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     text_input = torch.randn(1, 512)
#     img_input=torch.randn(1,512)
#     model = attention_cnn_fusion().to(device)
#     model.eval()
#     with torch.no_grad():
#         output = model(text_input.to(device),img_input.to(device))
#         print(output.shape)