"""
Policy: wraps the LLM for generation and log-prob computation.

Uses LoRA (PEFT) so only adapter weights are trained.
The frozen base model acts as the reference for KL — no copy needed.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType


class Policy:
    def __init__(
        self,
        model_name: str,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.0,
        device_map: str = "auto",
        torch_dtype=None,
    ):
        import torch as _torch
        dtype = torch_dtype or _torch.bfloat16

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        base = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device_map,
            max_memory={0: "83GiB"},
            trust_remote_code=True,
        )

        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        )
        self.model = get_peft_model(base, lora_cfg)
        self.model.print_trainable_parameters()
        self.model.train()

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompts: list[str],
        n_samples: int,
        temperature: float = 0.8,
        max_new_tokens: int = 200,
    ) -> list[list[str]]:
        """Return n_samples completions for each prompt (response only)."""
        self.model.eval()
        results = []

        with torch.no_grad():
            for prompt in prompts:
                inputs = self.tokenizer(
                    prompt, return_tensors="pt").to(self.device)
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    num_return_sequences=n_samples,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.3,
                )
                # import pdb; pdb.set_trace()
                input_len = inputs["input_ids"].shape[1]
                completions = [self.tokenizer.decode(out[inputs["input_ids"].shape[1]:], skip_special_tokens=True)for out in outputs]
                
                results.append(completions)
            # print(f"G samples are: {results}")

        self.model.train()
        self.model.gradient_checkpointing_enable()
        return results

    # ------------------------------------------------------------------
    # Log-prob computation 
    # ------------------------------------------------------------------

    def log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Per-token log probs for response tokens only.

        Returns tensor [batch, seq_len-1] with 0 on prompt positions.
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return self._token_log_probs(outputs.logits, input_ids, response_mask)

    def ref_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Log probs from the frozen base model (LoRA adapters disabled).
        lora base-with-adapters-off  """
        with self.model.disable_adapter():
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return self._token_log_probs(outputs.logits, input_ids, response_mask)

    @staticmethod
    def _token_log_probs(
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Shift of logit
        logits = logits[:, :-1, :]           # [B, L-1, V]
        labels = input_ids[:, 1:]            # [B, L-1]
        mask = response_mask[:, 1:]          # [B, L-1]

        lp = F.log_softmax(logits, dim=-1)
        token_lp = lp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # [B, L-1]
        return token_lp * mask  #mask-> zero out prompt tokens

    # ------------------------------------------------------------------
    # Tokenisation helper
    # ------------------------------------------------------------------

    def tokenize(
        self,
        prompt: str,
        completion: str,
        max_len: int = 2048,
    ) -> dict:
        """
        Tokenise prompt+completion and return a response_mask marking only
        the completion tokens (used to ignore prompt tokens in the loss).
        """
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full = self.tokenizer(
            prompt + completion,
            add_special_tokens=False,
            truncation=True,
            max_length=max_len,
        )
        input_ids = torch.tensor(full["input_ids"])
        response_mask = torch.zeros(len(input_ids))
        response_mask[len(prompt_ids):] = 1.0
        return {"input_ids": input_ids, "response_mask": response_mask}
