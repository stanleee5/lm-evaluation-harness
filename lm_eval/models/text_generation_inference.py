""" Text Generation Inference - python client
Example usage:
    python main.py --model tgi-client --model_args base_url=$URL --no_cache --tasks piqa

"""
import logging
from typing import List, Tuple

from tqdm import tqdm
from lm_eval.base import BaseLM

from text_generation import Client
from text_generation.types import Response


logger = logging.getLogger(__name__)


class TGIClient(BaseLM):
    def __init__(self, base_url, max_concurrency: int = 1):
        super().__init__()
        self.client = Client(base_url=base_url)
        self.max_concurrency = max_concurrency

    @property
    def eot_token_id(self):
        # Isn't used because we override loglikelihood, loglikelihood_rolling and greedy_until
        raise NotImplementedError()

    @property
    def max_length(self):
        # NOTE: Turn on truncation to avoid errors on long inputs.
        return 2048

    @property
    def max_gen_toks(self):
        return 512

    @property
    def batch_size(self):
        # Isn't used because we override loglikelihood, loglikelihood_rolling and greedy_until
        raise NotImplementedError()

    @property
    def device(self):
        # Isn't used because we override loglikelihood, loglikelihood_rolling and greedy_until
        raise NotImplementedError()

    def tok_encode(self, string: str):
        # Isn't used because we override loglikelihood, loglikelihood_rolling and greedy_until
        raise NotImplementedError()

    def tok_decode(self, tokens):
        # Isn't used because we override loglikelihood, loglikelihood_rolling and greedy_until
        raise NotImplementedError()

    def _preprocess_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]
        return context, continuation

    def loglikelihood(
        self, requests: List[Tuple[str, str]]
    ) -> List[Tuple[float, bool]]:
        def prefill_logprob(generated: Response) -> float:
            return sum(token.logprob for token in generated.details.prefill[1:])

        parameters = {
            "do_sample": False,
            "max_new_tokens": 1,
            "return_full_text": False,
            "decoder_input_details": True,
        }

        res = []
        for idx, (context, continuation) in enumerate(tqdm(requests)):
            context, continuation = self._preprocess_pair(context, continuation)
            concated = context + continuation

            generated_context = self.client.generate(context, **parameters)
            generated_concated = self.client.generate(concated, **parameters)

            logprob_context = prefill_logprob(generated_context)
            logprob_concated = prefill_logprob(generated_concated)

            logprob_continuation = logprob_concated - logprob_context
            res.append((logprob_continuation, False))
        return res

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError()

    def greedy_until(self, requests: List[Tuple[str, str]]) -> List[str]:
        if not requests:
            return []

        res = []

        for context, until in tqdm(requests):
            generated = self.client.generate(
                prompt=context,
                do_sample=False,
                stop=until,
            )
            res.append(generated.generated_text)
        return res

    def _model_call(self, inps):
        raise NotImplementedError()

    def _model_generate(self, context, max_length, eos_token_id):
        raise NotImplementedError()
