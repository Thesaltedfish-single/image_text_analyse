import torch
import torch.nn as nn
from transformers import BertModel
import torchvision
class img_txt_model(nn.Module):
    '''
    定义了一个文本-图像多模态分类的模型，是一个二分类模型
    '''
    def __init__(self):
        super().__init__()
        self.txt_model = BertModel.from_pretrained('bert-base-uncased')
        self.img_model = torchvision.models.densenet201(pretrained=True)
        self.t_linear = nn.Linear(768, 128)
        self.i_linear = nn.Linear(1000, 128)
        self.img_q = nn.Linear(128, 1)
        self.txt_q = nn.Linear(128, 1)
        self.fc = nn.Linear(128, 2)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask, image):
        img_out = self.img_model(image)
        img_out = self.i_linear(img_out)
        img_out = self.relu(img_out)
        img_weight = self.img_q(img_out)
        txt_out = self.txt_model(input_ids=input_ids, attention_mask=attention_mask)
        txt_out = txt_out.last_hidden_state[:,0,:]
        txt_out = txt_out.view(txt_out.shape[0], -1)
        txt_out = self.t_linear(txt_out)
        txt_out = self.relu(txt_out)
        txt_weight = self.txt_q(txt_out)
        last_out = img_weight * img_out + txt_weight * txt_out
        last_out = self.fc(last_out)
        return last_out
