import os
import math
import csv
import torch
import torchvision
import numpy as np
from model import *
import config as c
import datasets
import modules.Unet_common as common
from calculate_PSNR_SSIM import calculate_psnr, calculate_ssim
from tqdm import tqdm

def load(name, net, optim=None):
    state_dicts = torch.load(name, map_location="cpu")
    state_dict = state_dicts['net']
    # 'module.' prefix 제거
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    net.load_state_dict(new_state_dict)
    if optim is not None:
        try:
            optim.load_state_dict(state_dicts['opt'])
        except:
            print('Cannot load optimizer for some reason or other')


def tensor_to_image(tensor):
    np_img = tensor.detach().cpu().numpy()
    if np_img.ndim == 3:
        np_img = np_img.transpose(1, 2, 0)
    np_img = np.clip(np_img * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return np_img

def save_image(np_img, outdir, fname):
    os.makedirs(outdir, exist_ok=True)
    from PIL import Image
    Image.fromarray(np_img).save(os.path.join(outdir, fname))

def gauss_noise(shape, device):
    noise = torch.zeros(shape, device=device)
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape, device=device)
    return noise

def main():
    # 모델 경로 입력 받기
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='FP32 모델 checkpoint 경로')
    args = parser.parse_args()
    model_path = args.model

    # 모델명 추출 (확장자 및 경로 제외)
    model_name = os.path.splitext(os.path.basename(model_path))[0]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Model().to(device)
    init_model(net)
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=c.lr)
    load(model_path, net, optim)
    net.eval()

    dwt = common.DWT().to(device)
    iwt = common.IWT().to(device)

    # 결과 저장 폴더 경로 지정
    save_dirs = {
        "cover": os.path.join(c.IMAGE_PATH_cover, model_name),
        "secret": os.path.join(c.IMAGE_PATH_secret, model_name),
        "steg": os.path.join(c.IMAGE_PATH_steg, model_name),
        "secret_rev": os.path.join(c.IMAGE_PATH_secret_rev, model_name)
    }
    for d in save_dirs.values():
        os.makedirs(d, exist_ok=True)

    # csv 준비
    csv_name = f"{model_name}.csv"
    csv_path = os.path.join(os.getcwd(), csv_name)

    psnr_c_list, psnr_r_list = [], []
    ssim_c_list, ssim_r_list, ssim_avg_list = [], [], []
    img_names = []

    with torch.no_grad(), open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["img_name", "psnr_c", "psnr_r", "ssim_c", "ssim_r", "ssim_avg"])

        for i, (cover, secret) in tqdm(
            enumerate(datasets.testloader),
            total=len(datasets.testloader),
            desc="Evaluating",
        ):
            cover = cover.to(device)
            secret = secret.to(device)
            cover_input = dwt(cover)
            secret_input = dwt(secret)
            input_img = torch.cat((cover_input, secret_input), 1)

            # Forward
            output = net(input_img)
            output_steg = output.narrow(1, 0, 4 * c.channels_in)
            output_z = output.narrow(1, 4 * c.channels_in, output.shape[1] - 4 * c.channels_in)
            steg_img = iwt(output_steg)
            backward_z = gauss_noise(output_z.shape, device=device)

            # Backward
            output_rev = torch.cat((output_steg, backward_z), 1)
            backward_img = net(output_rev, rev=True)
            secret_rev = backward_img.narrow(1, 4 * c.channels_in, backward_img.shape[1] - 4 * c.channels_in)
            secret_rev = iwt(secret_rev)

            batch_size = cover.shape[0]
            for j in range(batch_size):
                img_index = i * batch_size + j
                img_name = f"{img_index + 1:05d}.png"
                img_names.append(img_name)

                cover_np = tensor_to_image(cover[j])
                secret_np = tensor_to_image(secret[j])
                steg_np = tensor_to_image(steg_img[j])
                secret_rev_np = tensor_to_image(secret_rev[j])

                save_image(cover_np, save_dirs["cover"], img_name)
                save_image(secret_np, save_dirs["secret"], img_name)
                save_image(steg_np, save_dirs["steg"], img_name)
                save_image(secret_rev_np, save_dirs["secret_rev"], img_name)

                # PSNR/SSIM 계산 (오류시 0)
                try:
                    psnr_c = calculate_psnr(
                        cover_np.astype(np.float32), steg_np.astype(np.float32)
                    )
                    psnr_r = calculate_psnr(
                        secret_np.astype(np.float32), secret_rev_np.astype(np.float32)
                    )
                    ssim_c = calculate_ssim(
                        cover_np.astype(np.float32), steg_np.astype(np.float32)
                    )
                    ssim_r = calculate_ssim(
                        secret_np.astype(np.float32), secret_rev_np.astype(np.float32)
                    )
                    ssim_avg = (ssim_c + ssim_r) / 2
                except Exception:
                    psnr_c = psnr_r = ssim_c = ssim_r = ssim_avg = 0

                # inf/nan 방지
                if np.isinf(psnr_c) or np.isnan(psnr_c):
                    psnr_c = 0
                if np.isinf(psnr_r) or np.isnan(psnr_r):
                    psnr_r = 0

                psnr_c_list.append(psnr_c)
                psnr_r_list.append(psnr_r)
                ssim_c_list.append(ssim_c)
                ssim_r_list.append(ssim_r)
                ssim_avg_list.append(ssim_avg)

                writer.writerow(
                    [
                        img_name,
                        f"{psnr_c:.6f}",
                        f"{psnr_r:.6f}",
                        f"{ssim_c:.6f}",
                        f"{ssim_r:.6f}",
                        f"{ssim_avg:.6f}",
                    ]
                )

        # 평균값 기록 (맨 마지막 줄)
        avg_psnr_c = np.mean(psnr_c_list)
        avg_psnr_r = np.mean(psnr_r_list)
        avg_ssim_c = np.mean(ssim_c_list)
        avg_ssim_r = np.mean(ssim_r_list)
        avg_ssim_avg = np.mean(ssim_avg_list)
        writer.writerow([
            "average",
            f"{avg_psnr_c:.6f}",
            f"{avg_psnr_r:.6f}",
            f"{avg_ssim_c:.6f}",
            f"{avg_ssim_r:.6f}",
            f"{avg_ssim_avg:.6f}",
        ])

    print(f"모든 결과가 {csv_path}에 저장되었습니다.")
    for key, val in save_dirs.items():
        print(f"[{key}] 이미지: {val}")

if __name__ == "__main__":
    main()