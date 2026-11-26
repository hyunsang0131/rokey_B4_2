"""(Colab, L4) 얼룩 3클래스(none/liquid/powder) 분류기 학습 → ONNX. spill_node 의 model_path 용.

데이터 폴더 구조 (얼룩 crop 이미지, 웹캠/RealSense 로 직접 촬영):
    spill_data/none/*.jpg  spill_data/liquid/*.jpg  spill_data/powder/*.jpg
    !python train_spill_classifier.py --data /content/drive/MyDrive/assist_arm/spill_data --out spill_cls.onnx
입력 규약 = spill_node: 1x3x64x64 RGB, 0~1, 클래스 순서 [none, liquid, powder]
"""
import argparse

import torch
import torch.nn as nn
from torchvision import datasets, models, transforms

ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True)
ap.add_argument('--out', default='spill_cls.onnx')
ap.add_argument('--epochs', type=int, default=15)
a = ap.parse_args()

CLASSES = ['none', 'liquid', 'powder']
tf = transforms.Compose([transforms.Resize((64, 64)), transforms.RandomHorizontalFlip(),
                         transforms.ColorJitter(0.2, 0.2, 0.1), transforms.ToTensor()])
ds = datasets.ImageFolder(a.data, transform=tf)
assert sorted(ds.classes) == sorted(CLASSES), ds.classes
remap = {ds.class_to_idx[c]: i for i, c in enumerate(CLASSES)}
ds.target_transform = lambda t: remap[t]
n_val = max(1, len(ds) // 5)
tr, va = torch.utils.data.random_split(ds, [len(ds) - n_val, n_val])
dl = torch.utils.data.DataLoader(tr, batch_size=64, shuffle=True)
dv = torch.utils.data.DataLoader(va, batch_size=64)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
m = models.mobilenet_v3_small(weights='DEFAULT')
m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 3)
m = m.to(dev)
opt = torch.optim.AdamW(m.parameters(), 1e-3)
for ep in range(a.epochs):
    m.train()
    for x, y in dl:
        opt.zero_grad()
        nn.functional.cross_entropy(m(x.to(dev)), y.to(dev)).backward()
        opt.step()
    m.eval()
    ok = n = 0
    with torch.no_grad():
        for x, y in dv:
            ok += (m(x.to(dev)).argmax(1).cpu() == y).sum().item()
            n += len(y)
    print(f'epoch {ep + 1} val acc {ok / n:.3f}')
m = m.cpu().eval()
torch.onnx.export(m, torch.zeros(1, 3, 64, 64), a.out, input_names=['x'], output_names=['logits'],
                  opset_version=17)
print('저장:', a.out)
