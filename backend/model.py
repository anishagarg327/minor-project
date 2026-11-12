import torch
import torch.nn as nn
import torchvision.transforms as tfs
import cv2
import numpy as np
from PIL import Image
import io

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_ffmpeg_exe():
    import shutil
    bin_path = shutil.which('ffmpeg')
    if bin_path:
        return bin_path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return 'ffmpeg'

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)

class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
    def forward(self, x):
        y = self.pa(x)
        return x * y

class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y

class Block(nn.Module):
    def __init__(self, conv, dim, kernel_size,):
        super(Block, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)

    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        res += x
        return res

class Group(nn.Module):
    def __init__(self, conv, dim, kernel_size, blocks):
        super(Group, self).__init__()
        modules = [Block(conv, dim, kernel_size) for _ in range(blocks)]
        modules.append(conv(dim, dim, kernel_size))
        self.gp = nn.Sequential(*modules)

    def forward(self, x):
        res = self.gp(x)
        res += x
        return res

class FFA(nn.Module):
    def __init__(self,gps,blocks,conv=default_conv):
        super(FFA, self).__init__()
        self.gps = gps
        self.dim = 64
        kernel_size = 3
        pre_process = [conv(3, self.dim, kernel_size)]
        assert self.gps==3
        self.g1 = Group(conv, self.dim, kernel_size,blocks=blocks)
        self.g2 = Group(conv, self.dim, kernel_size,blocks=blocks)
        self.g3 = Group(conv, self.dim, kernel_size,blocks=blocks)
        self.ca = nn.Sequential(*[
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dim*self.gps,self.dim//16,1,padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim//16, self.dim*self.gps, 1, padding=0, bias=True),
            nn.Sigmoid()
            ])
        self.palayer = PALayer(self.dim)

        post_process = [
            conv(self.dim, self.dim, kernel_size),
            conv(self.dim, 3, kernel_size)]

        self.pre = nn.Sequential(*pre_process)
        self.post = nn.Sequential(*post_process)

    def forward(self, x1):
        x = self.pre(x1)
        res1 = self.g1(x)
        res2 = self.g2(res1)
        res3 = self.g3(res2)
        w = self.ca(torch.cat([res1,res2,res3],dim=1))
        w = w.view(-1,self.gps, self.dim)[:,:,:,None,None]
        out = w[:,0,::] * res1 + w[:,1,::] * res2 + w[:,2,::] * res3
        out = self.palayer(out)
        x = self.post(out)
        return x + x1

class VideoProcessor:
    def __init__(self, model_path):
        self.net = FFA(gps=3, blocks=12)
        self.net = nn.DataParallel(self.net)
        self.net = self.net.to(device)
        ckp = torch.load(model_path, map_location=device, weights_only=False)
        self.net.load_state_dict(ckp['model'])
        self.net.eval()

    def process_frame(self, frame, is_video=False):
        orig_h, orig_w = frame.shape[:2]
        # Video frames on CPU need smaller dimension for fast processing
        max_dim = (240 if is_video else 360) if device == 'cpu' else 1280
        if max(orig_h, orig_w) > max_dim:
            scale = max_dim / max(orig_h, orig_w)
            w, h = (int(orig_w * scale) // 4) * 4, (int(orig_h * scale) // 4) * 4
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        else:
            w, h = orig_w, orig_h
        frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        transform = tfs.Compose([
            tfs.ToTensor(),
            tfs.Normalize(mean=[0.64, 0.6, 0.58], std=[0.14, 0.15, 0.152])
        ])
        frame = transform(frame).unsqueeze(0).to(device)

        with torch.no_grad():
            dehazed = self.net(frame)

        dehazed = dehazed.squeeze(0).cpu()
        dehazed = dehazed.permute(1, 2, 0).numpy() * 255
        dehazed = dehazed.astype(np.uint8)
        dehazed = cv2.cvtColor(dehazed, cv2.COLOR_RGB2BGR)
        if (w, h) != (orig_w, orig_h):
            dehazed = cv2.resize(dehazed, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)

        return dehazed

    def process_image(self, image_path):
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError("Unable to read image")

        dehazed = self.process_frame(frame, is_video=False)
        return dehazed

    def process_video(self, video_bytes):
        import subprocess
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_input = os.path.join(temp_dir, 'input.mp4')
            temp_output = os.path.join(temp_dir, 'output.mp4')

            with open(temp_input, 'wb') as f:
                f.write(video_bytes)

            cap = cv2.VideoCapture(temp_input)
            fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                cap.release()
                raise ValueError("Could not read frames from video file.")

            # Sample keyframes for fast, reliable processing on CPU/cloud
            max_keyframes = 8 if device == 'cpu' else 30
            step = max(1, total_frames // max_keyframes)

            frame_idx = 0
            saved_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % step == 0 and saved_count < max_keyframes:
                    processed = self.process_frame(frame, is_video=True)
                    frame_path = os.path.join(temp_dir, f'frame_{saved_count:04d}.png')
                    cv2.imwrite(frame_path, processed)
                    saved_count += 1
                frame_idx += 1

            cap.release()

            if saved_count == 0:
                raise ValueError("No frames were extracted from video.")

            # Preserve original video playback duration
            duration = max(1.0, total_frames / fps)
            target_fps = max(1.0, saved_count / duration)

            combine_cmd = [
                get_ffmpeg_exe(), '-y',
                '-framerate', str(target_fps),
                '-i', os.path.join(temp_dir, 'frame_%04d.png'),
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-crf', '23',
                '-preset', 'veryfast',
                '-movflags', '+faststart',
                temp_output
            ]
            subprocess.run(combine_cmd, check=True)

            with open(temp_output, 'rb') as f:
                processed_video = f.read()

            return processed_video
