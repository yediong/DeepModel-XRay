from transformers import T5Model, BertModel  # 导入T5和BERT模型
import torchvision.models as models          # 导入torchvision中的模型（如VGG16）
import torch                                 
from torchinfo import summary                # 导入torchinfo的summary用于结构和参数展示

# T5-small
model_t5 = T5Model.from_pretrained("t5-small")  # 加载预训练的T5-small模型
# 用torchinfo.summary输出模型结构和参数统计（注意输入需包含input_ids和decoder_input_ids）
summary(
    model_t5,
    input_data={
        "input_ids": torch.ones(1, 8, dtype=torch.long),
        "decoder_input_ids": torch.ones(1, 8, dtype=torch.long)
    }
)
print("\n唯一参数量统计：")
total_params = sum(p.numel() for p in model_t5.parameters())  # 统计唯一参数量（不重复统计共享参数）
print(f"Total params: {total_params:,}")

# VGG16
model_vgg = models.vgg16(pretrained=True)  # 加载预训练的VGG16模型
# 用torchinfo.summary输出VGG16结构和参数统计，输入为1张3通道224x224图片
summary(model_vgg, input_size=(1, 3, 224, 224))

# Bert-base-uncased
model_bert = BertModel.from_pretrained("bert-base-uncased")  # 加载预训练的BERT-base-uncased模型
# 用torchinfo.summary输出BERT结构和参数统计，输入为input_ids
summary(
    model_bert,
    input_data={"input_ids": torch.ones(1, 8, dtype=torch.long)}
)