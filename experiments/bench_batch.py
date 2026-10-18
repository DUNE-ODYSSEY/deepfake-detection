"""Largest batch size (from the candidates) whose AMP train step fits in GPU memory
with headroom, so a run never spills into Windows shared memory (10x slower).

Prints the chosen batch size on the last line, or 0 if none fit.
Usage: python -m experiments.bench_batch --model efficientnet_b4 --candidates 32 16
"""
import argparse

import torch

from src.models.factory import create_model, model_img_size


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--candidates", nargs="+", type=int, default=[32, 16])
    p.add_argument("--max-gb", type=float, default=5.3)
    args = p.parse_args()
    size = model_img_size(args.model)
    for bs in args.candidates:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = create_model(args.model).cuda().train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler()
        try:
            for _ in range(3):
                x = torch.randn(bs, 3, size, size, device="cuda")
                y = torch.randint(0, 2, (bs,), device="cuda").float()
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda"):
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x), y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_reserved() / 2**30
        except torch.cuda.OutOfMemoryError:
            peak = float("inf")
        print(f"{args.model} batch {bs}: peak reserved {peak:.2f} GB", flush=True)
        del model, opt
        if peak <= args.max_gb:
            print(bs)
            return
    print(0)


if __name__ == "__main__":
    main()
