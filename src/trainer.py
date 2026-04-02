import os
import time
from collections import deque
from logging import getLogger

import torch
from torch.cuda.amp import autocast, GradScaler

from src.models.model import evaluate

logger = getLogger()


def reload_model_optimizer(args, model, optimizer):
    model_path = os.path.join(args.dump_path, "model.pt")
    optimizer_path = os.path.join(args.dump_path, "optimizer.pt")
    scaler_path = os.path.join(args.dump_path, "scaler.pt")
    
    if os.path.isfile(model_path):
        logger.info("resuming from existing model")
        if args.device == "cuda":
            reloaded = torch.load(model_path)
        else:
            reloaded = torch.load(model_path, map_location=torch.device(args.device))
        model.load_state_dict(reloaded)
    if os.path.isfile(optimizer_path):
        logger.info("resuming from existing optimizer")
        if args.device == "cuda":
            reloaded = torch.load(optimizer_path)
        else:
            reloaded = torch.load(optimizer_path, map_location=torch.device(args.device))
        optimizer.load_state_dict(reloaded)
    
    # Load scaler for AMP if available
    scaler = None
    if args.use_amp and os.path.isfile(scaler_path):
        logger.info("resuming from existing GradScaler")
        if args.device == "cuda":
            scaler = torch.load(scaler_path)
    
    return scaler


def train(model, args, loader, optim, test_dataset, current_best_loss=None, scaler=None):
    """
    Train the model with optional Automatic Mixed Precision (AMP).
    
    AMP benefits:
    - Reduces VRAM usage by ~50% (allows larger batch sizes)
    - Speeds up training on modern GPUs (Tensor Cores)
    - Maintains accuracy with loss scaling
    """
    best_loss = current_best_loss or float("inf")
    curr_loss = 0
    
    # Initialize GradScaler for AMP if enabled
    use_amp = args.use_amp and args.device == "cuda"
    if use_amp and scaler is None:
        scaler = GradScaler(init_scale=2.**10)  # Start with reasonable scale
    
    for step in range(args.max_steps):
        if step % 100 == 0:
            t0 = time.time()
        
        batch = loader.next()
        batch = [t.to(args.device) for t in batch]
        X, Y = batch[0], batch[1]

        # Forward pass with AMP
        if use_amp:
            with autocast(dtype=torch.float16):
                _, loss, _ = model(X, Y)
            # Scale loss and backward
            scaler.scale(loss).backward()
            # Unscales gradients and calls optimizer.step()
            scaler.unscale_(optim)
            # Clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # Optimizer step
            scaler.step(optim)
            # Update scale for next iteration
            scaler.update()
            # Clear gradients
            optim.zero_grad(set_to_none=True)
        else:
            # Standard training (no AMP)
            _, loss, _ = model(X, Y)
            model.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
        
        curr_loss += loss.item()

        # Logging
        if (step + 1) % 100 == 0:
            t1 = time.time()
            log_msg = f"step {step + 1} | loss {loss.item():.4f} | steps time {(t1-t0)*1000:.2f}ms"
            
            # Log GPU memory usage periodically
            if args.device == "cuda" and step % 500 == 0:
                allocated = torch.cuda.memory_allocated(0) / (1024*1024)
                reserved = torch.cuda.memory_reserved(0) / (1024*1024)
                log_msg += f" | VRAM: {allocated:.0f}/{reserved:.0f}MB"
                
                # Log AMP scale if enabled
                if use_amp:
                    log_msg += f" | AMP scale: {scaler.get_scale():.1f}"
            
            logger.info(log_msg)
            
        if (step + 1) % args.num_eval_steps == 0:
            train_loss = curr_loss / args.num_eval_steps
            test_loss = evaluate(model, test_dataset, args.device, batch_size=100, max_batches=10)
            logger.info(f"step {step + 1} train loss: {train_loss} test loss: {test_loss}")
            
            if args.save_best and test_loss < best_loss:
                model_path = os.path.join(args.dump_path, "model.pt")
                optimizer_path = os.path.join(args.dump_path, "optimizer.pt")
                scaler_path = os.path.join(args.dump_path, "scaler.pt")
                
                torch.save(model.state_dict(), model_path)
                torch.save(optim.state_dict(), optimizer_path)
                
                # Save AMP scaler state
                if use_amp:
                    torch.save(scaler.state_dict(), scaler_path)
                
                logger.info(f"test loss {test_loss} is the best so far, saved model to {model_path}")
                best_loss = test_loss
            
            curr_loss = 0

    if not args.save_best:
        model_path = os.path.join(args.dump_path, "model.pt")
        optimizer_path = os.path.join(args.dump_path, "optimizer.pt")
        scaler_path = os.path.join(args.dump_path, "scaler.pt")
        
        torch.save(model.state_dict(), model_path)
        torch.save(optim.state_dict(), optimizer_path)
        
        # Save AMP scaler state
        if use_amp:
            torch.save(scaler.state_dict(), scaler_path)

    return best_loss, scaler
