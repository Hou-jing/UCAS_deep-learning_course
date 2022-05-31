import torch
import torch.nn as nn

criterion = nn.BCELoss()#默认是求均值，数据需要是浮点型数据
pre=torch.tensor([0.1,0.2,0.3,0.4]).float()
tar=torch.tensor([0,0,0,1]).float()
l=criterion(pre,tar)
print('二分类交叉熵损失函数计算（均值）',l)


pre=torch.tensor([0.2,0.8,0.4,0.1,0.9]).float()
tar=torch.tensor([0,1,0,0,1]).float()

pre=torch.tensor([0.1,0.2,0.3,0.4]).float()
tar=torch.tensor([0,0,0,1]).float()
criterion = nn.BCELoss(reduction="sum")#求和
l=criterion(pre,tar)
print('二分类交叉熵损失函数计算（求和）',l)

loss=nn.BCELoss(reduction="none")#reduction="none"得到的是loss向量#对每一个样本求损失
l=loss(pre,tar)
print('每个样本对应的loss',l)
criterion2=nn.CrossEntropyLoss()
import numpy as np
pre1=torch.tensor([np.log(20),np.log(40),np.log(60),np.log(80)]).float()
# soft=nn.Softmax(dim=0)
# pre=soft(pre).float()#bs*label_nums
pre1=pre1.reshape(1,4)
tar=torch.tensor([3])
loss2=criterion2(pre1,tar)
print('多分类交叉熵损失函数pre1条件下',loss2)

pre2=torch.tensor([np.log(10),np.log(30),np.log(50),np.log(90)]).float()
pre2=pre2.reshape(1,4)
tar=torch.tensor([3])
loss2=criterion2(pre2,tar)
print('多分类交叉熵损失函数pre2条件下',loss2)
