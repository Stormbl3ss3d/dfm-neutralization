import os
import torch
from pathlib import Path
from transformers import AutoTokenizer
from omegaconf import OmegaConf

from model import Transformer
from flow_matching.path.scheduler import PolynomialConvexScheduler
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.solver import MixtureDiscreteEulerSolver
from flow_matching.utils import ModelWrapper

class ConditionalWrappedModel(ModelWrapper):
    def __init__(self, model, prompt_ids, target_mask):
        super().__init__(model=model)
        self.prompt_ids = prompt_ids
        self.target_mask = target_mask

    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
        # 1. Clamp the inputs so the model always sees the clean prompt
        x = torch.where(self.target_mask, x, self.prompt_ids)
        
        # 2. Get model predictions
        probs = torch.softmax(self.model(x_t=x, time=t).float(), -1)
        
        # 3. Force the output probabilities of the prompt tokens to be 100% 
        # so the solver doesn't try to "update" them with noise.
        probs[~self.target_mask] = 0.0
        prompt_indices = self.prompt_ids[~self.target_mask]
        probs[~self.target_mask, prompt_indices] = 1.0
        
        return probs

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load latest checkpoint automatically
    checkpoints = list(Path("outputs").rglob("checkpoint.pth"))
    if not checkpoints:
        raise ValueError("No checkpoints found in outputs/")
    latest_ckpt = max(checkpoints, key=os.path.getmtime)
    print(f"Loading checkpoint: {latest_ckpt}")
    loaded_state = torch.load(latest_ckpt, map_location=device, weights_only=True)

    # 2. Initialize Tokenizer & Model (matching our training config)
    tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
    cfg = OmegaConf.create({
        "hidden_size": 384, "cond_dim": 128, "length": 128,
        "n_blocks": 6, "n_heads": 6, "dropout": 0.0
    })
    model = Transformer(config=cfg, vocab_size=tokenizer.vocab_size, masked=False).to(device)
    model.load_state_dict(loaded_state["model"])
    model.eval()

    # 3. Prepare the Input (We'll use one of your replace-only examples)
    biased_text = 'the court ruled that indecent exposure only relates to exposure of the genitals, adding that whilst mooning was a "disgusting" and "demeaning" act to engage in, and had in addition taken place in the presence of a minor,'
    
    prompt_text = f"Biased: {biased_text} Neutral:"
    prompt_enc = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    
    # Structure: [CLS] Biased: <src> Neutral: 
    input_ids = [tokenizer.cls_token_id] + prompt_enc
    target_start = len(input_ids)
    max_length = 128

    # Create the Target Mask (False for prompt, True for the area to generate)
    target_mask = [False] * target_start + [True] * (max_length - target_start)
    target_mask = torch.tensor(target_mask, dtype=torch.bool, device=device).unsqueeze(0)

    # Pad the prompt out to 128 length for clamping
    prompt_ids = input_ids + [tokenizer.pad_token_id] * (max_length - len(input_ids))
    prompt_ids = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)

    # Initialize the sequence with pure uniform noise (our source distribution)
    x_init = torch.randint(size=(1, max_length), high=tokenizer.vocab_size, device=device)
    x_init = torch.where(target_mask, x_init, prompt_ids) # Stamp the clear prompt on top

    # 4. Run the Discrete Flow Solver
    wrapped_model = ConditionalWrappedModel(model, prompt_ids, target_mask)
    path = MixtureDiscreteProbPath(scheduler=PolynomialConvexScheduler(n=1.0))
    solver = MixtureDiscreteEulerSolver(
        model=wrapped_model, 
        path=path, 
        vocabulary_size=tokenizer.vocab_size
    )

    print("\nStarting generation (unmasking from t=0 to t=1)...")
    with torch.no_grad():
        sample = solver.sample(
            x_init=x_init,
            step_size=1 / 128, # 128 sampling steps
            verbose=False,
            time_grid=torch.tensor([0.0, 1.0])
        )

    # 5. Extract and decode the generated target
    generated_ids = sample[0][target_start:]
    result_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    print("\n" + "="*50)
    print(f"BIASED:  {biased_text}")
    print(f"NEUTRAL: {result_text}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
