import time
from datetime import datetime
from pathlib import Path

import torch
from torch.profiler import profile, record_function, ProfilerActivity

from pointcept.engines.train import Trainer
from pointcept.models.point_prompt_training import PointPromptTrainingLoRA, configure_adamw_lora


class LoRATrainer(Trainer):
    def build_optimizer(self):
        if self.cfg.optimizer.pop("type") != "AdamW":
            raise NotImplementedError
        return configure_adamw_lora(self.model, **self.cfg.optimizer)

    def train(self):
        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            profile_memory=True, record_shapes=True
        ) 
        prof.__enter__()
        try:
            super(LoRATrainer, self).train()
        except torch.cuda.OutOfMemoryError as e:
            prof.__exit__(None, None, None) ## ??
            print(e)
            now = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            save_to = (Path(self.cfg.save_path) / f"trace_{now}.json").resolve()
            print(f"Exporting crash profile trace to: {save_to}")
            prof.export_chrome_trace(str(save_to))
            raise 
        else:
            prof.__exit__(None, None, None)

