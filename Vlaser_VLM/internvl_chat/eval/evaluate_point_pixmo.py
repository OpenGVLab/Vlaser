import argparse
import itertools
import json
import os
import random
import time
from functools import partial
import cv2
import imageio
import numpy as np
import torch
from decord import VideoReader, cpu
from internvl.model import load_model_and_tokenizer
from internvl.train.dataset import build_transform, dynamic_preprocess
import numpy as np
import re
import time
from math import sqrt
from functools import partial
from PIL import Image, ImageDraw
from tqdm import tqdm


def point_in_radius(hw, pred, targets, radius=25.0):
   
    for tgt in targets:
        dist = sqrt((pred[0] - tgt[0]) ** 2 + (pred[1] - tgt[1]) ** 2)
        if dist <= radius :
            return True
    return False


def load_gt_points(path):
    gt_dict = {}
    with open(path) as f:
        for line in f:
            item = json.loads(line.strip())
            image = item['image']
            for conv in item['conversations']:
                if conv['from'] == 'gpt':
                    expr_match = re.search(r'<ref>(.*?)</ref>', conv['value'])
                    points = re.findall(r'\[\s*(\d+)\s*,\s*(\d+)\s*\]', conv['value'])
                    if expr_match and points:
                        expr = expr_match.group(1)
                        coords = [(int(x), int(y)) for x, y in points]
                        gt_dict[(image, expr)] = coords
    return gt_dict


class PixmoPointsDataset(torch.utils.data.Dataset):
    
    def __init__(self, test_path, prompt, input_size=224, dynamic_image_size=False,
                 use_thumbnail=False, max_num=6):
        self.datas = []
        with open(test_path) as f:
            for line in f:
                item = json.loads(line.strip())
                image_path = item['image']
                w, h = item['width'], item['height']
                for conv in item['conversations']:
                    if conv['from'] == 'human':
                        sentence = conv['value']
                        match = re.search(r'<ref>(.*?)</ref>', sentence)
                        if not match:
                            continue
                        expr = match.group(1)
                        self.datas.append({
                            'image': os.path.join('/mnt/petrelfs/share_data/zhangtianyi1/pixmo-points-images-eval', image_path),
                            'expression': expr,
                            'width': w,
                            'height': h,
                        })
        self.prompt = prompt
        self.input_size = input_size
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.max_num = max_num
        self.transform = build_transform(is_train=False, input_size=input_size)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = self.datas[idx]
        image = Image.open(data['image']).convert('RGB')
        if self.dynamic_image_size:
            images = dynamic_preprocess(image, image_size=self.input_size,
                                        use_thumbnail=self.use_thumbnail,
                                        max_num=self.max_num)
        else:
            images = [image]
        pixel_values = [self.transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)

        return {
            'text': self.prompt.format(data['expression']),
            'pixel_values': pixel_values,
            'image': os.path.basename(data['image']),
            'expression': data['expression'],
            'hw': (data['height'], data['width']),
        }


def collate_fn(batches, tokenizer):
    pixel_values = torch.cat([_['pixel_values'] for _ in batches], dim=0)
    texts = [_['text'] for _ in batches]
    images = [_['image'] for _ in batches]
    expressions = [_['expression'] for _ in batches]
    hws = [_['hw'] for _ in batches]
    return pixel_values, texts, images, expressions, hws


def evaluate_pixmo_model(args):
    model, tokenizer = load_model_and_tokenizer(args)
    prompt = ' Please provide the referring points of {}.'
    prompt0 = ' You are InternVL. Your task is to locate several points in the given image according to the task descriptions. Your answer should be formatted as \"<point>[[x1, y1], [x2, y2],...]</point>\". The point coordinates are normalized to integers between 0 and 1000. Return the answer in the point format directly.'
    prompt = prompt0 + prompt
    image_size = model.config.force_image_size or model.config.vision_config.image_size
    use_thumbnail = model.config.use_thumbnail

    dataset = PixmoPointsDataset(
        test_path=args.data_path,
        prompt=prompt,
        input_size=image_size,
        dynamic_image_size=args.dynamic,
        use_thumbnail=use_thumbnail,
        max_num=args.max_num
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,
        num_workers=args.num_workers,
        collate_fn=partial(collate_fn, tokenizer=tokenizer)
    )

    model.cuda()
    model.eval()
    pred_results = []

    for index, (pixel_values, questions, images, expressions, hws) in enumerate(tqdm(dataloader)):
        pixel_values = pixel_values.to(torch.bfloat16).cuda()
        generation_config = dict(
            num_beams=args.num_beams,
            max_new_tokens=100,
            min_new_tokens=1,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
        )

        pred = model.chat(
            tokenizer=tokenizer,
            pixel_values=pixel_values,
            question=questions[0],
            generation_config=generation_config,
            verbose=False
        )
        pred_results.append({
            'answer': pred,
            'image': images[0],
            'expression': expressions[0],
            'hw': hws[0],
        })
    
    gt_points = load_gt_points(args.data_path)

    PATTERN = re.compile(r'\[\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\]')
    correct = total = 0

    for result in pred_results:
        key = (result['image'], result['expression'])
        if key not in gt_points:
            continue
        gt = gt_points[key]
        pred_points = PATTERN.findall(result['answer'])
        pred_points = [(float(x), float(y)) for x, y in pred_points]

        for p in pred_points:
            if point_in_radius(result['hw'], p, gt, radius=args.radius):
                correct += 1
        if len(pred_points) >= 1:
            total += len(pred_points)
        else: 
            total += 1

    acc = correct / total if total > 0 else 0
    print(f'Precision @ {args.radius}px: {acc:.4f}')

    os.makedirs(args.out_dir, exist_ok=True)
    timestamp = time.strftime('%y%m%d_%H%M%S')
    out_path = os.path.join(args.out_dir, f'pixmo_eval.json')
    json.dump(pred_results, open(out_path, 'w'), indent=2)
    out_acc_path = os.path.join(args.out_dir, f'pixmo_eval_acc.json')
    json.dump(f'correct: {correct}, total: {total}, acc: {acc:.4f}', open(out_acc_path, 'w'), indent=2)
    print(f'Predictions saved to: {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default="OpenGVLab/Vlaser-2B", help='Path to InternVL model checkpoint')
    parser.add_argument('--data-path', type=str, default="/mnt/petrelfs/share_data/zhangtianyi1/internvlembodied/pixmo_points_json/data_0.jsonl", help='Path to PixMo-Points jsonl file')
    parser.add_argument('--out-dir', type=str, default='./eval_output', help='Directory to save output')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--num-beams', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--dynamic', type=bool, default=True)
    parser.add_argument('--max-num', type=int, default=6)
    parser.add_argument('--radius', type=float, default=25.0, help='Evaluation tolerance radius in pixels')
    parser.add_argument('--load-in-8bit', action='store_true')
    parser.add_argument('--load-in-4bit', action='store_true')
    parser.add_argument('--auto', action='store_true')
    args = parser.parse_args()
    if args.checkpoint[-1] == '/':
        args.checkpoint = args.checkpoint[:-1]
    args.out_dir = os.path.join(args.out_dir, args.checkpoint.split('/')[-2] + '_' + args.checkpoint.split('/')[-1])
    evaluate_pixmo_model(args)
