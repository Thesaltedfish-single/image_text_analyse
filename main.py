from model import img_txt_model
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
import numpy as np
import pandas as pd
import os 
import matplotlib.pyplot as plt
import argparse
from PIL import Image
from sklearn.model_selection import train_test_split

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print ('device:',device)
Dataset_path='./dataset/data/'
filled_image_path='./dataset/填充.jpg'

def txt_emb(txt,tokenzier):
    result=tokenzier(txt,return_tensors='pt',padding=True,truncation=True)
    input_ids=result['input_ids'].to(device)
    attention_mask=result['attention_mask'].to(device)
    return input_ids,attention_mask 

class Dataset():
    def __init__(self, images, texts, labels, token):
        self.images = images
        self.texts = texts
        self.labels = labels
        self.input_ids, self.attention_masks = txt_emb(self.texts, token)
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        # print('index', idx)
        img = self.images[idx]
        text = self.texts[idx]
        label = self.labels[idx]
        input_id = self.input_ids[idx]
        attention_mask = self.attention_masks[idx]
        return img, text, label, input_id, attention_mask
    
def train_process(model, epoch_num, optimizer, train_dataloader, valid_dataloader, train_count, valid_count):
    '''
    训练模型，在每一个epoch后评估其在验证集上的性能
    '''
    Loss_f = nn.CrossEntropyLoss()#交叉熵损失
    train_acc = []
    valid_acc = []
    for epoch in range(epoch_num):
        loss = 0.0
        train_cor_count = 0
        valid_cor_count = 0
        for b_idx, (img, des, target, idx, mask) in enumerate(train_dataloader):
            img, mask, idx, target = img.to(device), mask.to(device), idx.to(device), target.to(device)
            output = model(idx, mask, img)
            optimizer.zero_grad()
            loss = Loss_f(output, target)
            loss.backward() #反向传播计算梯度
            optimizer.step() #更新模型参数
            pred = output.argmax(dim=1)
            train_cor_count += int(pred.eq(target).sum())
        train_acc.append(train_cor_count / train_count)
        for img, des, target, idx, mask in valid_dataloader:
            img, mask, idx, target = img.to(device), mask.to(device), idx.to(device), target.to(device)
            output = model(idx, mask, img)
            pred = output.argmax(dim=1)
            valid_cor_count += int(pred.eq(target).sum())
        valid_acc.append(valid_cor_count / valid_count)
        print('Train Epoch: {}, Train_Loss: {:.4f}, Train Accuracy: {:.4f}, Valid Accuracy: {:.4f}'.format(epoch + 1,
                                                                                                           loss.item(),
                                                                                                           train_cor_count / train_count,
                                                                                                           valid_cor_count / valid_count))
    plt.plot(train_acc, label="train_accuracy")
    plt.plot(valid_acc, label="valid_accuracy")
    plt.title(model.__class__.__name__)
    plt.xlabel("Epoch")
    plt.xticks(range(epoch_num), range(1, epoch_num + 1))
    plt.ylabel("Accuracy")
    plt.ylim(ymin=0, ymax=1)
    plt.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--epoch_num', default=4, type=int)
    parser.add_argument('--pretrained_model_name', default="bert-base-chinese", type=str)
    args=parser.parse_args()
    pretrained_model_name=args.pretrained_model_name
    
    #加载模型
    model=img_txt_model().to(device)
    tokenizer=BertTokenizer.from_pretrained(pretrained_model_name)
    pretrained_model=BertModel.from_pretrained(pretrained_model_name)
    #设置超参
    lr=args.lr
    epoch_num=args.epoch_num
    #设置优化器
    optimizer=optim.Adam(model.parameters(),lr=lr)

    images=[]
    texts=[]
    labels=[]
    '''
    1:发生诈骗
    0：没有发生诈骗
    '''
    is_fruad={1.0:1,0.0:0}
    train_df=pd.read_csv('./dataset/data_train.csv',sep='\t')
    #从数据集中读取数据
    for i in range(len(train_df)):
       print(train_df.columns)
       guid=train_df.iloc[i]['guid']
       label=train_df.iloc[i]['label']
       image_path=Dataset_path+str(guid)+'.jpg'
       text_path=Dataset_path+str(guid)+'.txt'
       if not os.path.exists(image_path):
           img=Image.open(filled_image_path)
       else:
           img=Image.open(image_path)
       text=open(text_path,'r',encoding='utf-8').read()
       img = img.resize((224, 224), Image.Resampling.LANCZOS)
       img = np.asarray(img, dtype='float32')
       img = np.transpose(img, (2, 0, 1))
       images.append(img)
       texts.append(text)
       labels.append(is_fruad[label])
    
    #划分数据集开始训练
    img_txt_pairs = [(images[i], texts[i]) for i in range(len(texts))]
    X_train, X_valid, tag_train, tag_valid = train_test_split(img_txt_pairs, labels, test_size=0.2, random_state=1458, shuffle=True)
    image_train, txt_train = [X_train[i][0] for i in range(len(X_train))], [X_train[i][1] for i in range(len(X_train))]
    image_valid, txt_valid = [X_valid[i][0] for i in range(len(X_valid))], [X_valid[i][1] for i in range(len(X_valid))]
    train_dataset = Dataset(image_train, txt_train, tag_train, tokenizer)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    valid_dataset = Dataset(image_valid, txt_valid, tag_valid, tokenizer)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=16, shuffle=True)
    train_process(model, epoch_num, optimizer, train_dataloader, valid_dataloader, len(X_train), len(X_valid))


if __name__ == '__main__':
    main()




