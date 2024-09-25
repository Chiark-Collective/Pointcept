from pointcept.engines.train import Trainer
from pointcept.models.point_prompt_training import PointPromptTrainingLoRA, configure_adamw_lora


class LoRATrainer(Trainer):
    def build_optimizer(self):
        if self.cfg.optimizer.pop("type") != "AdamW":
            raise NotImplementedError
        return configure_adamw_lora(self.model, **self.cfg.optimizer)
