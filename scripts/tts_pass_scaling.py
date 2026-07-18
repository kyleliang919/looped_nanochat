import types, torch
from nanochat.checkpoint_manager import load_model
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.loss_eval import evaluate_bpb
from nanochat.common import compute_init, compute_cleanup, print0
ddp, rank, local_rank, world, dev = compute_init("cuda")
model,tok,meta=load_model("base",dev,phase="eval",model_tag="looped-d24"); model.eval()
token_bytes=get_token_bytes(device=dev)
BS,SEQ,STEPS=32,2048,80  # 4 GPUs x 32 x 2048 x 80 ~= 21M tokens per K
for K in [1,2,3,4,5,6,8]:
    model.forward = types.MethodType(
        lambda self,idx,targets=None,loss_reduction="mean",_K=K: self.forward_iterated(idx,_K,targets,loss_reduction),
        model)
    loader=tokenizing_distributed_data_loader_bos_bestfit(tok,BS,SEQ,split="val",device=dev)
    bpb=evaluate_bpb(model,loader,STEPS,token_bytes)
    print0(f"K={K}  val_bpb={bpb:.5f}")
compute_cleanup()
