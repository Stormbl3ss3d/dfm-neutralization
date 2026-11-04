import os
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from omegaconf import OmegaConf

from model import Transformer
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.solver import MixtureDiscreteEulerSolver
from flow_matching.utils import ModelWrapper

class GuidedWrappedModel(ModelWrapper):
    def __init__(self, model, prompt_ids, target_mask, classifier, word_embeddings, guidance_scale):
        super().__init__(model=model)
        self.prompt_ids = prompt_ids
        self.target_mask = target_mask
        self.classifier = classifier
        self.word_embeddings = word_embeddings
        self.guidance_scale = guidance_scale

    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
        x_masked = torch.where(self.target_mask, x, self.prompt_ids)
        
        with torch.enable_grad():
            logits = self.model(x_t=x_masked, time=t).float()
            logits.requires_grad_(True)
            
            probs_for_grad = torch.softmax(logits, dim=-1)
            
            soft_embeds = torch.matmul(probs_for_grad, self.word_embeddings)
            class_logits = self.classifier(inputs_embeds=soft_embeds).logits
            
            bias_score = class_logits[0, 1]
            gradients = torch.autograd.grad(outputs=bias_score, inputs=logits)[0]
            
        # Normalize gradients for stable logit updates
        grad_norm = torch.norm(gradients, dim=-1, keepdim=True) + 1e-8
        normalized_gradients = gradients / grad_norm
        
        steered_logits = logits.detach() - (self.guidance_scale * normalized_gradients)
        steered_probs = torch.softmax(steered_logits, dim=-1)
        
        steered_probs[~self.target_mask] = 0.0
        prompt_indices = self.prompt_ids[~self.target_mask]
        steered_probs[~self.target_mask, prompt_indices] = 1.0
        
        return steered_probs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    # nargs='+' allows us to pass a space-separated list of strings
    parser.add_argument("--input_text", type=str, nargs='+', required=True) 
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--guidance_scale", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
    
    print("Loading bias classifier for gradient steering...")
    classifier_path = os.path.expanduser("~/dfm-neutralization/checkpoints/detector")
    classifier = AutoModelForSequenceClassification.from_pretrained(classifier_path).to(device)
    classifier.eval()
    word_embeddings = classifier.get_input_embeddings().weight

    print(f"Loading generative checkpoint: {args.checkpoint_path}")
    loaded_state = torch.load(args.checkpoint_path, map_location=device, weights_only=True)
    cfg = OmegaConf.create({
        "hidden_size": 768, "cond_dim": 128, "length": 128,
        "n_blocks": 6, "n_heads": 12, "dropout": 0.0
    })
    model = Transformer(config=cfg, vocab_size=tokenizer.vocab_size, masked=False).to(device)
    model.load_state_dict(loaded_state["model"])
    model.eval()

    print(f"\nProcessing {len(args.input_text)} sentences with guidance_scale {args.guidance_scale}...\n")
    print("="*60)

    # Loop through each sentence provided in the arguments
    for idx, text in enumerate(args.input_text):
        prompt_text = f"Biased: {text} Neutral:"
        prompt_enc = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        
        input_enc = tokenizer(text, add_special_tokens=False)["input_ids"]
        target_budget = len(input_enc) + 5
        
        prefix_ids = [tokenizer.cls_token_id] + prompt_enc + [tokenizer.sep_token_id]
        target_start_idx = len(prefix_ids)
        target_end_idx = target_start_idx + target_budget
        max_length = 128
        
        target_mask = [0] * max_length
        for i in range(target_start_idx, min(target_end_idx, max_length - 1)):
            target_mask[i] = 1
            
        prompt_ids = prefix_ids + [tokenizer.pad_token_id] * target_budget + [tokenizer.sep_token_id]
        pad_len = max_length - len(prompt_ids)
        if pad_len > 0:
            prompt_ids = prompt_ids + [tokenizer.pad_token_id] * pad_len
        else:
            prompt_ids = prompt_ids[:max_length]
            
        target_mask = torch.tensor(target_mask, dtype=torch.bool, device=device).unsqueeze(0)
        prompt_ids = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)

        x_init = torch.randint(size=(1, max_length), high=tokenizer.vocab_size, device=device)
        x_init = torch.where(target_mask, x_init, prompt_ids) 

        wrapped_model = GuidedWrappedModel(
            model, prompt_ids, target_mask, classifier, word_embeddings, args.guidance_scale
        )
        path = MixtureDiscreteProbPath(scheduler=PolynomialConvexScheduler(n=1.0))
        solver = MixtureDiscreteEulerSolver(
            model=wrapped_model, 
            path=path, 
            vocabulary_size=tokenizer.vocab_size
        )
        
        # Let the framework's native solver handle the loop
        with torch.no_grad():
            sample = solver.sample(
                x_init=x_init,
                step_size=1.0 / args.steps, 
                verbose=False,
                time_grid=torch.linspace(0.0, 1.0, args.steps + 1, device=device)
            )
            
        generated_ids = sample[0][target_mask[0]]
        result_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        print(f"[{idx+1}/{len(args.input_text)}]")
        print(f"BIASED:  {text}")
        print(f"NEUTRAL: {result_text}")
        print("-" * 60)

if __name__ == "__main__":
    main()