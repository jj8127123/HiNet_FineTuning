import os
import math
import torch
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from model import *
import config as c
import datasets
import modules.Unet_common as common

# 모델명(확장자 없는 부분만)
model_name = os.path.splitext(os.path.basename(c.suffix))[0]

# 결과 저장 폴더 및 파일명
result_dir = "psnr_results"
os.makedirs(result_dir, exist_ok=True)
csv_path = os.path.join(result_dir, f"psnr_log_{model_name}.csv")
png_path = os.path.join(result_dir, f"psnr_plot_{model_name}.png")

# 네트워크 및 옵티마이저 초기화
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
net = Model()
net.cuda()
init_model(net)
net = torch.nn.DataParallel(net, device_ids=c.device_ids)
params_trainable = list(filter(lambda p: p.requires_grad, net.parameters()))
optim = torch.optim.Adam(params_trainable, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)

def load(name, net, optim):
    state_dicts = torch.load(name)
    network_state_dict = {k: v for k, v in state_dicts['net'].items() if 'tmp_var' not in k}
    net.load_state_dict(network_state_dict)
    try:
        optim.load_state_dict(state_dicts['opt'])
    except:
        print('Cannot load optimizer for some reason or other')

def gauss_noise(shape):
    noise = torch.zeros(shape).cuda()
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape).cuda()
    return noise

def computePSNR(origin, pred):
    origin = np.array(origin)
    pred = np.array(pred)
    origin = origin.astype(np.float32)
    pred = pred.astype(np.float32)
    mse = np.mean((origin - pred) ** 2)
    if mse < 1e-10:
        return 100
    return 10 * math.log10(255.0 ** 2 / mse)

load(c.MODEL_PATH + c.suffix, net, optim)
net.eval()
dwt = common.DWT().to(device)
iwt = common.IWT().to(device)

# 전체 데이터 개수 예측 (DataLoader, Dataset 구조에 따라 조정)
total_samples = len(datasets.testloader.dataset)

# 결과 기록
psnr_records = []

start_time = time.time()

with torch.no_grad():
    for i, (cover, secret) in enumerate(datasets.testloader):
        cover = cover.to(device)
        secret = secret.to(device)
        cover_input = dwt(cover)
        secret_input = dwt(secret)
        input_img = torch.cat((cover_input, secret_input), dim=1)

        output = net(input_img)
        output_steg = output.narrow(1, 0, 4 * c.channels_in)
        output_z = output.narrow(1, 4 * c.channels_in, output.shape[1] - 4 * c.channels_in)
        steg_img = iwt(output_steg)
        backward_z = gauss_noise(output_z.shape)

        output_rev = torch.cat((output_steg, backward_z), dim=1)
        backward_img = net(output_rev, rev=True)
        secret_rev = backward_img.narrow(1, 4 * c.channels_in, backward_img.shape[1] - 4 * c.channels_in)
        secret_rev = iwt(secret_rev)

        batch_size = cover.size(0)
        for j in range(batch_size):
            idx = i * batch_size + j

            # 이미지 저장 (모델명 하위 폴더로 분리하려면 아래 코드 참고)
            torchvision.utils.save_image(cover[j], f"{c.IMAGE_PATH_cover}{idx:05d}.png")
            torchvision.utils.save_image(secret[j], f"{c.IMAGE_PATH_secret}{idx:05d}.png")
            torchvision.utils.save_image(steg_img[j], f"{c.IMAGE_PATH_steg}{idx:05d}.png")
            torchvision.utils.save_image(secret_rev[j], f"{c.IMAGE_PATH_secret_rev}{idx:05d}.png")

            psnr_r = computePSNR(secret[j].cpu(), secret_rev[j].cpu())
            psnr_c = computePSNR(cover[j].cpu(), steg_img[j].cpu())
            psnr_records.append({"index": idx, "PSNR_r": psnr_r, "PSNR_c": psnr_c})

        # 진행률 출력 (10회마다, 또는 마지막 배치)
        if (i % 10 == 0) or (i == len(datasets.testloader) - 1):
            processed = (i + 1) * batch_size
            percent = (processed / total_samples) * 100
            elapsed = time.time() - start_time
            print(f"[{processed:5d}/{total_samples}] {percent:5.1f}% 완료 | 경과시간: {elapsed:.1f}초")

# CSV 저장
df = pd.DataFrame(psnr_records)
df.to_csv(csv_path, index=False)

# 시각화
plt.figure(figsize=(10, 5))
plt.plot(df["index"], df["PSNR_r"], label="PSNR_r (Secret)")
plt.plot(df["index"], df["PSNR_c"], label="PSNR_c (Cover)")
plt.xlabel("Sample Index")
plt.ylabel("PSNR (dB)")
plt.title(f"PSNR Evaluation: {model_name}")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(png_path)
plt.close()
