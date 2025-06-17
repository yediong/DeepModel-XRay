from transformers import T5Model
import torchvision.models as models
from transformers import BertModel, BertTokenizer
import torch

# 加载 t5-small 预训练模型
model1 = T5Model.from_pretrained("t5-small")
# 打印模型结构
print("="*60)
print("T5-small 模型完整结构:")
print(model1)

# 加载预训练的 VGGNet 模型
model2 = models.vgg16(pretrained=True)
# 打印模型结构
print("VGG 模型完整结构:")
print(model2)

# 加载BERT基础模型（uncased版本）和分词器
model3 = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
print("Bert 模型完整结构:")
print(model3)