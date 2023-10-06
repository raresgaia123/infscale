import torch, time, os, random
from torch.utils.data import DataLoader, Dataset
import deepspeed
from deepspeed.pipe import PipelineModule
from torchvision.models.resnet import resnet50, ResNet50_Weights

num_batches = 100
num_classes = 1000
batch_size = 64
image_w = 224
image_h = 224


class MyDataset(Dataset):
    def __init__(self, img_list, num_class=10):
        super(MyDataset, self).__init__()
        self.img_list = img_list
        self.num_classes = num_class

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = self.img_list[idx]
        return img, random.randint(1, self.num_classes)


def flat_func(x):
    return torch.flatten(x, 1)


local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "2"))

deepspeed.init_distributed()

net = resnet50(num_classes=num_classes)
layers = [
    net.conv1,
    net.bn1,
    net.relu,
    net.maxpool,
    net.layer1,
    net.layer2,
    net.layer3,
    net.layer4,
    net.avgpool,
    flat_func,
    net.fc,
]

# generating inputs
inputs = [
    torch.randn(3, image_w, image_h, dtype=torch.float)
    for i in range(num_batches * batch_size)
]
dataset = MyDataset(inputs, num_class=num_classes)

model = PipelineModule(layers, 2)
model_engine, optimizer, dataloader, _ = deepspeed.initialize(
    model=model, config="ds_config.json", training_data=dataset
)
model = model_engine

tik = time.time()
for i in range(num_batches):
    outputs = model.eval_batch(iter(dataloader), compute_loss=False)

tok = time.time()
print(f"{tok - tik}, {(num_batches * batch_size) / (tok - tik)}")
