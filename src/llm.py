from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.prompts import with_final_answer_cue
from src.types import Messages


class LocalLLM:
    def __init__(self, config_path: str | Path = "config.yaml") -> None:
        config = yaml.safe_load(Path(config_path).read_text())["llm"]
        self.batch_size = config["batch_size"]
        self.max_new_tokens = config["max_new_tokens"]

        self.tokenizer = AutoTokenizer.from_pretrained(config["model_name"], padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quantization = (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            if config["load_in_4bit"]
            else None
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            config["model_name"],
            quantization_config=quantization,
            dtype=torch.float16,
            device_map="auto",
        ).eval()

    def _encode(self, batch: list[Messages], continue_final: bool = False) -> dict[str, torch.Tensor]:
        prompts = [
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=not continue_final,
                continue_final_message=continue_final,
            )
            for messages in batch
        ]
        return self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)

    def _option_token_ids(self, option: str) -> list[int]:
        # After a cue like "Final answer:" the answer is usually a space-prefixed token,
        # while a bare assistant turn starts without one; score every surface form.
        variants = {option, option.capitalize(), f" {option}", f" {option.capitalize()}"}
        return sorted({self.tokenizer.encode(v, add_special_tokens=False)[0] for v in variants})

    @torch.inference_mode()
    def _score_options(
        self, batch: list[Messages], options: list[str], continue_final: bool
    ) -> list[str]:
        option_ids = [self._option_token_ids(option) for option in options]
        flat_ids = [tid for ids in option_ids for tid in ids]
        if len(flat_ids) != len(set(flat_ids)):
            raise ValueError("Options must begin with distinct tokens")

        predictions: list[str] = []
        for start in range(0, len(batch), self.batch_size):
            inputs = self._encode(batch[start:start + self.batch_size], continue_final)
            # Project only the final hidden state through lm_head; full-sequence logits
            # over a ~150k vocabulary would exhaust T4 memory.
            hidden = self.model.model(**inputs).last_hidden_state[:, -1]
            logits = self.model.lm_head(hidden).float()
            scores = torch.stack([logits[:, ids].max(dim=-1).values for ids in option_ids], dim=-1)
            predictions.extend(options[i] for i in scores.argmax(dim=-1).tolist())
        return predictions

    def classify(self, batch: list[Messages], options: list[str]) -> list[str]:
        return self._score_options(batch, options, continue_final=False)

    @torch.inference_mode()
    def generate(self, batch: list[Messages]) -> list[str]:
        outputs: list[str] = []
        for start in range(0, len(batch), self.batch_size):
            inputs = self._encode(batch[start:start + self.batch_size])
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            new_tokens = generated[:, inputs["input_ids"].shape[1]:]
            outputs.extend(self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
        return outputs

    def reason_then_classify(
        self, batch: list[Messages], options: list[str]
    ) -> tuple[list[str], list[str]]:
        rationales = self.generate(batch)
        continued = [with_final_answer_cue(m, r) for m, r in zip(batch, rationales)]
        return self._score_options(continued, options, continue_final=True), rationales
