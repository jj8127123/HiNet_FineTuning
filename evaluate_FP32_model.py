import argparse
import os
import csv
import torch
import numpy as np
from hinet import Hinet
import config as c
from torch.utils.data import DataLoader
import datasets
import modules.Unet_common as common
from calculate_PSNR_SSIM import calculate_psnr, calculate_ssim
from tqdm import tqdm
from torchvision import transforms as T


def load_model(model_path: str, device: torch.device) -> torch.nn.Module:
    """Load a full-precision model checkpoint."""
    model = Hinet()
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    img = tensor.detach().cpu().numpy()
    if img.ndim == 3:
        img = img.transpose(1, 2, 0)
    img = np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return img


def save_img(np_img: np.ndarray, save_dir: str, img_name: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    from PIL import Image
    Image.fromarray(np_img).save(os.path.join(save_dir, img_name))


def evaluate(model_path: str) -> str:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation will be performed on device: {device}")
    net = load_model(model_path, device)

    dwt = common.DWT().to(device)
    iwt = common.IWT().to(device)

    transform_val = T.Compose([
        T.CenterCrop(c.cropsize_val),
        T.ToTensor(),
    ])
    dataset = datasets.HinetDataset(
        c.VAL_PATH, c.VAL_COVER_PATH, transform_val, c.format_val
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=False)

    psnr_c_list, psnr_r_list = [], []
    ssim_c_list, ssim_r_list, ssim_avg_list = [], [], []
    img_names = [f"{i+1:04d}.png" for i in range(len(dataset))]

    model_name = os.path.splitext(os.path.basename(model_path))[0]
    save_dirs = {
        "cover": os.path.join(c.IMAGE_PATH_cover, model_name),
        "secret": os.path.join(c.IMAGE_PATH_secret, model_name),
        "steg": os.path.join(c.IMAGE_PATH_steg, model_name),
        "secret_rev": os.path.join(c.IMAGE_PATH_secret_rev, model_name),
    }
    for d in save_dirs.values():
        os.makedirs(d, exist_ok=True)

    with torch.no_grad():
        for idx, (secret, cover) in tqdm(
            enumerate(loader), total=len(loader), desc="Evaluating FP32"
        ):
            secret = secret.to(device)
            cover = cover.to(device)
            cover_input = dwt(cover)
            secret_input = dwt(secret)
            input_img = torch.cat((cover_input, secret_input), 1)

            output = net(input_img)
            output_steg = output[:, : 4 * c.channels_in]
            output_z = output[:, 4 * c.channels_in :]
            steg_img = iwt(output_steg)
            backward_z = torch.randn_like(output_z).to(device)

            output_rev = torch.cat((output_steg, backward_z), 1)
            backward_img = net(output_rev, rev=True)
            secret_rev = iwt(backward_img[:, 4 * c.channels_in :])

            cover_np = tensor_to_image(cover[0])
            steg_np = tensor_to_image(steg_img[0])
            secret_np = tensor_to_image(secret[0])
            secret_rev_np = tensor_to_image(secret_rev[0])

            img_name = img_names[idx]
            save_img(cover_np, save_dirs["cover"], img_name)
            save_img(secret_np, save_dirs["secret"], img_name)
            save_img(steg_np, save_dirs["steg"], img_name)
            save_img(secret_rev_np, save_dirs["secret_rev"], img_name)

            cover_np = cover_np.astype(np.float32)
            steg_np = steg_np.astype(np.float32)
            secret_np = secret_np.astype(np.float32)
            secret_rev_np = secret_rev_np.astype(np.float32)

            psnr_c = calculate_psnr(cover_np, steg_np)
            psnr_r = calculate_psnr(secret_np, secret_rev_np)
            if np.isinf(psnr_c) or np.isnan(psnr_c):
                psnr_c = 0
            if np.isinf(psnr_r) or np.isnan(psnr_r):
                psnr_r = 0

            ssim_c = calculate_ssim(cover_np, steg_np)
            ssim_r = calculate_ssim(secret_np, secret_rev_np)
            ssim_avg = (ssim_c + ssim_r) / 2

            psnr_c_list.append(psnr_c)
            psnr_r_list.append(psnr_r)
            ssim_c_list.append(ssim_c)
            ssim_r_list.append(ssim_r)
            ssim_avg_list.append(ssim_avg)

    avg_psnr_c = sum(psnr_c_list) / len(psnr_c_list)
    avg_psnr_r = sum(psnr_r_list) / len(psnr_r_list)
    avg_ssim_c = sum(ssim_c_list) / len(ssim_c_list)
    avg_ssim_r = sum(ssim_r_list) / len(ssim_r_list)
    avg_ssim_avg = sum(ssim_avg_list) / len(ssim_avg_list)

    csv_dir = "./result_csv/"
    os.makedirs(csv_dir, exist_ok=True)
    csv_name = f"{model_name}.csv"
    csv_path = os.path.join(csv_dir, csv_name)

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["img_name", "psnr_c", "psnr_r", "ssim_c", "ssim_r", "ssim_avg"])
        for name, pc, pr, sc, sr, sa in zip(
            img_names, psnr_c_list, psnr_r_list, ssim_c_list, ssim_r_list, ssim_avg_list
        ):
            writer.writerow([name, f"{pc:.6f}", f"{pr:.6f}", f"{sc:.6f}", f"{sr:.6f}", f"{sa:.6f}"])
        writer.writerow([
            "average",
            f"{avg_psnr_c:.6f}",
            f"{avg_psnr_r:.6f}",
            f"{avg_ssim_c:.6f}",
            f"{avg_ssim_r:.6f}",
            f"{avg_ssim_avg:.6f}",
        ])
    print(f"Saved evaluation results to {csv_path}")
    return csv_path


def main():
    parser = argparse.ArgumentParser(description="Evaluate FP32 model PSNR/SSIM & save images")
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    args = parser.parse_args()
    csv_path = evaluate(args.model)
    print(csv_path)


if __name__ == "__main__":
    main()