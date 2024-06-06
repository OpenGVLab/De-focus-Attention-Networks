# --------------------------------------------------------
# Modified By Mzero
# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import torch
import torch.distributed as dist
from torch._six import inf
from timm.utils import ModelEma as ModelEma
import numpy
import matplotlib.pyplot as plt
from matplotlib.ticker import SymmetricalLogLocator
import io
import PIL
from torchvision.transforms import ToTensor


def load_checkpoint_ema(config, model, optimizer, lr_scheduler, loss_scaler, logger, model_ema: ModelEma=None):
    logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    
    if 'model' in checkpoint:
        msg = model.load_state_dict(checkpoint['model'], strict=False)
        logger.info(f"resuming model: {msg}")
    else:
        logger.warning(f"No 'model' found in {config.MODEL.RESUME}! ")

    if model_ema is not None:
        if 'model_ema' in checkpoint:
            msg = model_ema.ema.load_state_dict(checkpoint['model_ema'], strict=False)
            logger.info(f"resuming model_ema: {msg}")
        else:
            logger.warning(f"No 'model_ema' found in {config.MODEL.RESUME}! ")

    max_accuracy = 0.0
    max_accuracy_ema = 0.0
    if not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        config.freeze()
        if 'scaler' in checkpoint:
            loss_scaler.load_state_dict(checkpoint['scaler'])
        logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
        if 'max_accuracy' in checkpoint:
            max_accuracy = checkpoint['max_accuracy']
        if 'max_accuracy_ema' in checkpoint:
            max_accuracy_ema = checkpoint['max_accuracy_ema']

    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy, max_accuracy_ema


def load_pretrained_ema(config, model, logger, model_ema: ModelEma=None):
    logger.info(f"==============> Loading weight {config.MODEL.PRETRAINED} for fine-tuning......")
    checkpoint = torch.load(config.MODEL.PRETRAINED, map_location='cpu')
    
    if 'model' in checkpoint:
        msg = model.load_state_dict(checkpoint['model'], strict=False)
        logger.warning(msg)
        logger.info(f"=> loaded 'model' successfully from '{config.MODEL.PRETRAINED}'")
    else:
        logger.warning(f"No 'model' found in {config.MODEL.PRETRAINED}! ")

    if model_ema is not None:
        if 'model_ema' in checkpoint:
            msg = model_ema.ema.load_state_dict(checkpoint['model_ema'], strict=False)
            logger.warning(msg)
            logger.info(f"=> loaded 'model_ema' successfully from '{config.MODEL.PRETRAINED}'")
        else:
            logger.warning(f"No 'model_ema' found in {config.MODEL.PRETRAINED}! ")

    del checkpoint
    torch.cuda.empty_cache()


def save_checkpoint_ema(config, epoch, model, max_accuracy, optimizer, lr_scheduler, loss_scaler, logger, model_ema: ModelEma=None, max_accuracy_ema=None):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'scaler': loss_scaler.state_dict(),
                  'epoch': epoch,
                  'config': config}
    
    if model_ema is not None:
        save_state.update({'model_ema': model_ema.ema.state_dict(),
            'max_accuray_ema': max_accuracy_ema})

    save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")

    if isinstance(epoch, int):
        to_del = epoch - config.save_ckpt_num
        old_ckpt = os.path.join(config.OUTPUT, ('ckpt_epoch_%s.pth' % to_del))
        if os.path.exists(old_ckpt):
            os.remove(old_ckpt)


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth') and not ('best' in ckpt)]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def ampscaler_get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(),
                                                        norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True, model=None, dtype=None):
        if dtype == torch.float16:
            self._scaler.scale(loss).backward(create_graph=create_graph)
            if update_grad:
                if clip_grad is not None:
                    assert parameters is not None
                    self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                    norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
                else:
                    self._scaler.unscale_(optimizer)
                    norm = ampscaler_get_grad_norm(parameters)
                # self.print_nan_grad(model, norm)
                self._scaler.step(optimizer)
                self._scaler.update()
            else:
                norm = None
        elif dtype == torch.bfloat16:
            loss.backward(create_graph=create_graph)
            if update_grad:
                if clip_grad is not None:
                    assert parameters is not None
                    norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
                else:
                    norm = ampscaler_get_grad_norm(parameters)
                # if torch.distributed.get_rank() == 0:
                #     self.print_nan_grad(model, norm)
                optimizer.step()
            else:
                norm = None
        return norm
    
    def print_nan_grad(self, model, norm):
        # calculate grad in each para:
        print_grad = False
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if torch.isnan(param.grad).any():
                    print_grad = True
                    print(f"NaN grad found in {name}")
                    if torch.isnan(param).any():
                        print(f"NaA found in {name}")
                    if torch.isinf(param).any():
                        print(f"Inf found in {name}")
                elif torch.isinf(param.grad).any():
                    print_grad = True
                    print(f"Inf found in {name}")
                    if torch.isnan(param).any():
                        print(f"NaA found in {name}")
                    if torch.isinf(param).any():
                        print(f"Inf found in {name}")
        # import pdb; pdb.set_trace()
        if print_grad or norm > 100:
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    g_norm = torch.norm(param.grad)
                    g_mean = torch.mean(param.grad)
                    g_std = torch.std(param.grad)
                    mean = torch.mean(param)
                    std = torch.std(param)
                    print(f"{name}: grad mean={g_mean.item()}, grad std={g_std.item()}, grad norm={g_norm.item()}")
                    # print(f"{name}: mean={mean.item()}, std={std.item()}")
            import pdb; pdb.set_trace()

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)
        
def draw_offset(idx, samples, offsets, name, tblogger):  #offsets: (bsz, num_heads, N1, N2,  2)
    B = samples.shape[0]
    num_heads = offsets.shape[1]
    os.makedirs(f'./figs/{name}', exist_ok=True)
    H = samples.shape[2]//offsets.shape[2]
    W = samples.shape[3]//offsets.shape[3]

    for i in range(B):  #print multiple images
        img = samples[i].permute(1,2,0).cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min())
        for h in range(num_heads):
            plt.imshow(img)
            plt.title(f'Block {idx}, Image {i}, Head {h}')
            plt.axis('off')
            for x in range(offsets.shape[2]):
                for y in range(offsets.shape[3]):
                    x_pos = round((offsets[i,h,x,y,0]*H).item()) 
                    y_pos = round((offsets[i,h,x,y,1]*W).item())
                    plt.arrow(y*W+W//2,x*H+H//2,y_pos,x_pos, width=1, head_width=2, head_length=6, fc='red', ec='red')
            plt.savefig(f'./figs/{name}/block{idx}_image{i}_head{h}')

            buf = io.BytesIO()
            plt.savefig(buf, format='jpeg')
            buf.seek(0)
            tb_img = ToTensor()(PIL.Image.open(buf))
            tblogger.add_image(f'{name}/block{idx}_image{i}_head{h}', tb_img)

            plt.clf()
        if i == 2:
            break
    return

def draw_grad(images, model, input, tblogger=None, name='grad_map',args=None,cls_token_num=0):
    input = input.detach().requires_grad_(True)
    # import pdb;pdb.set_trace()
    if input.dim() == 4:
        output, _, x_vis = model(input)
        B, h, w, C = output.shape
        _, _, H, W = images.shape

        for i in range(h):
            for j in range(w):
                select_pix = x_vis[0, :, i, j]
                select_pix_norm = select_pix.norm()

                input.retain_grad = True
                select_pix_norm.backward(retain_graph=True)

                grad_map = input.grad[0].norm(dim=2, keepdim=True)
                grad_map = grad_map.view(h, 1, w, 1).repeat(1, H//h, 1, W//w).view(1, H, W)
                grad_map = (grad_map) / (grad_map.max())
                img = (images[0] - images[0].min()) / (images[0].max() - images[0].min())
                grad_map = (0.4+grad_map*0.6) * img
                
                # if j%5 == 0 and i%5 == 0:
                #     plt.imshow(grad_map.permute(1,2,0).cpu().numpy())
                #     plt.axis('off')
                #     plt.savefig(f'./figs/test/{i}_{j}')
                #     plt.clf()

                if tblogger is not None:
                    tblogger.add_image(f"{name}/x={i}, y={j}", grad_map)
                
                input.grad.zero_()
    elif input.dim() == 3:
        output = model(input)
        if isinstance(output,tuple):
            output, x_vis = output 
        B, seq_len, C = output.shape
        _, _, W, H = images.shape
        # assert W==H
        w = round((seq_len-cls_token_num)**0.5)  # For cls_token=32 here
        
        avg_grad_map = 0
        n_imgs = 32
        for pic in range(n_imgs):
            # select_token = output[pic,90,:]
            # select_token_norm = select_token.norm()
            # input.retain_grad = True
            # select_token_norm.backward(retain_graph=True)
            # grad_map = input.grad[pic].norm(dim=1, keepdim=False)
            # # grad_map = (grad_map) / (grad_map.max())
            # grad_map = grad_map[:-cls_token_num]
            # grad_map = grad_map.view(w, 1, w, 1).repeat(1, W//w, 1, W//w).view(1, W, W)
            # grad_map = (grad_map) / (grad_map.max())
            # img = (images[pic] - images[pic].min()) / (images[pic].max() - images[pic].min())
            # grad_map = (0.4+grad_map*0.6) * img
            # if tblogger is not None:
            #     tblogger.add_image(f"{name}/Point(6,6)", grad_map)
            # input.grad.zero_()

            # for i in range(cls_token_num):
            i = cls_token_num - 1
            select_token = output[pic,w**2+i,:]  # x_vis: [B, img_token+cls_token, embed_dim]
            select_token_norm = select_token.norm()
            input.retain_grad = True
            select_token_norm.backward(retain_graph=True)
            grad_map = input.grad[pic].norm(dim=1, keepdim=False)
            grad_map = (grad_map) / (grad_map.max())
            grad_map = grad_map[:-cls_token_num]
            grad_map = grad_map.view(w, 1, w, 1).repeat(1, W//w, 1, W//w).view(1, W, W)

            avg_grad_map += grad_map

            # grad_map = (grad_map) / (grad_map.max())
            # img = (images[pic] - images[pic].min()) / (images[pic].max() - images[pic].min())
            # grad_map = (0.2+grad_map*0.8) * img
            # if tblogger is not None:
            #     tblogger.add_image(f"{name}/Pic{pic}-token[{i}]", grad_map)
            input.grad.zero_()
        
        if tblogger is not None:
            avg_grad_map = avg_grad_map / n_imgs
            avg_grad_map = avg_grad_map / avg_grad_map.max()
            tblogger.add_image(f"{name}/token[{i}]", avg_grad_map)
    return output

def draw_total_grad(images, blocks, input, tblogger=None, name='grad_map',args=None,cls_token_num=0):
    input = input.detach().requires_grad_(True)
    # import pdb;pdb.set_trace()
    if input.dim() == 4:
        output, x_vis = blocks[0](input)
        for blk in blocks[1:]:
            output, x_vis = blk(output)
        B, h, w, C = output.shape
        _, _, H, W = images.shape

        for i in range(h):
            for j in range(w):
                select_pix = x_vis[0, :, i, j]
                select_pix_norm = select_pix.norm()
                input.retain_grad = True
                select_pix_norm.backward(retain_graph=True)

                grad_map = input.grad[0].norm(dim=2, keepdim=True)
                grad_map = grad_map.view(h, 1, w, 1).repeat(1, H//h, 1, W//w).view(1, H, W)
                grad_map = (grad_map) / (grad_map.max())
                img = (images[0] - images[0].min()) / (images[0].max() - images[0].min())
                grad_map = (0.4+grad_map*0.6) * img
                
                if tblogger is not None:
                    tblogger.add_image(f"{name}/x={i}, y={j}", grad_map)
                input.grad.zero_()
    elif input.dim() == 3:
        output = blocks[0](input)
        if isinstance(output,tuple):
            output = output[0]
            for blk in blocks[1:]:
                output, x_vis = blk(output)
        else:
            for blk in blocks[1:]:
                output = blk(output)
        B, seq_len, C = output.shape
        _, _, W, H = images.shape
        # assert W==H
        w = round((seq_len-cls_token_num)**0.5)
        
        for pic in range(32):
            # import pdb;pdb.set_trace()
            # select_token = output[pic,90,:]      # 90th token = patch(6,6) // (x_vis in final layer or output?)
            # select_token_norm = select_token.norm()
            # input.retain_grad = True
            # select_token_norm.backward(retain_graph=True)
            # grad_map = input.grad[pic].norm(dim=1, keepdim=False)
            # grad_map = grad_map[:-cls_token_num]
            # grad_map = grad_map.view(w, 1, w, 1).repeat(1, W//w, 1, W//w).view(1, W, W)
            # grad_map = (grad_map) / (grad_map.max())
            # img = (images[pic] - images[pic].min()) / (images[pic].max() - images[pic].min())
            # grad_map = (0.4+grad_map*0.6) * img
            # if tblogger is not None:
            #     tblogger.add_image(f"{name}/Pic{pic}-Point(6,6)", grad_map)
            # input.grad.zero_()
                
            for i in range(cls_token_num):
                select_token = output[pic,w**2+i,:]  # x_vis: [B, img_token+cls_token, embed_dim]
                select_token_norm = select_token.norm()
                input.retain_grad = True
                select_token_norm.backward(retain_graph=True)
                grad_map = input.grad[pic].norm(dim=1, keepdim=False)
                # grad_map = (grad_map) / (grad_map.max())
                grad_map = grad_map[:-cls_token_num]
                grad_map = grad_map.view(w, 1, w, 1).repeat(1, W//w, 1, W//w).view(1, W, W)
                grad_map = (grad_map) / (grad_map.max())
                img = (images[pic] - images[pic].min()) / (images[pic].max() - images[pic].min())
                grad_map = (0.2+grad_map*0.8) * img
                if tblogger is not None:
                    tblogger.add_image(f"{name}/Pic{pic}-token[{i}]", grad_map)
                input.grad.zero_()
    return output

def draw_grad_and_offset(images, model, input, tblogger=None, name='grad_map',args=None):
    input = input.detach().requires_grad_(True)
    output, offsets, x_vis = model(input)
    B, h, w, C = output.shape
    _, _, H, W = images.shape
    # import pdb;pdb.set_trace()
    for i in range(h):
        for j in range(w):
            select_pix = x_vis[0, :, i, j]
            select_pix_norm = select_pix.norm()
            input.retain_grad = True
            select_pix_norm.backward(retain_graph=True)
            
            grad_map = input.grad[0].norm(dim=2, keepdim=True)
            grad_map = grad_map.view(h, 1, w, 1).repeat(1, H//h, 1, W//w).view(1, H, W)
            grad_map = (grad_map) / (grad_map.max())
            img = (images[0] - images[0].min()) / (images[0].max() - images[0].min())
            grad_map = (0.4+grad_map*0.6) * img

            grad_map = grad_map.permute(1,2,0).cpu().numpy()
            plt.imshow(grad_map)
            plt.axis('off')
            for p in range(offsets.shape[1]):
                x_pos = round((offsets[0,p,i,j,0]*H//h).item()) 
                y_pos = round((offsets[0,p,i,j,1]*W//w).item()) 
                plt.arrow(j*W//w+W//w//2,i*H//h+H//h//2,y_pos,x_pos, width=1, head_width=2, head_length=6, fc='red', ec='red')
            # if j%5 == 0 and i%5 == 0:
            #     plt.savefig(f'./figs/test/{i}_{j}')
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            tb_img = ToTensor()(PIL.Image.open(buf))
            
            if tblogger is not None:
                tblogger.add_image(f"{name}/x={i}, y={j}, all head", tb_img)
            plt.clf()
            input.grad.zero_()
    return output


def plot_omega(init, learned):
    plt.subplot(2, 1, 1)
    plt.scatter(torch.arange(1, init.shape[1]+1), init[0], s=1, label='init_x')
    plt.scatter(torch.arange(1, learned.shape[1]+1), learned[0], s=1, label='learned_x')
    plt.legend()
    plt.yscale('symlog')
    # locator = SymmetricalLogLocator(linthresh=0.1, base=10, subs=torch.arange(1,10))
    # plt.gca().yaxis.set_major_locator(SymmetricalLogLocator(linthresh=0.1, base=10, subs=torch.arange(1,10)))
    # plt.gca().yaxis.set_minor_locator(SymmetricalLogLocator(linthresh=0.1, base=10, subs=torch.arange(1,10)))
    # plt.minorticks_on()
    plt.yticks(torch.range(-6, 6, 0.2))
    plt.grid()
    # plt.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
    # plt.grid(which='minor', linestyle='--', linewidth='0.5', color='gray', alpha=0.5)
    # plt.grid(which='minor', color='gray', linewidth='0.5', linestyle='--')
    plt.xlabel('index')
    plt.ylabel('value')

    plt.subplot(2, 1, 2)
    plt.scatter(torch.arange(1, init.shape[1]+1), init[1], s=1, label='init_y')
    plt.scatter(torch.arange(1, learned.shape[1]+1), learned[1], s=1, label='learned_y')
    plt.legend()
    plt.yscale('symlog')
    plt.gca().yaxis.set_major_locator(SymmetricalLogLocator(linthresh=0.1, base=10))
    plt.gca().yaxis.set_minor_locator(SymmetricalLogLocator(linthresh=0.1, base=10))
    plt.minorticks_on()
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
    plt.grid(which='minor', linestyle='--', linewidth='0.5', color='gray', alpha=0.5)
    plt.xlabel('index')
    plt.ylabel('value')

    plt.savefig('figs/omega.png', dpi=300)