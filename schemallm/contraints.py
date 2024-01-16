import logging
from typing import Union, Any, Type, List
from pydantic import BaseModel
from transformers import PreTrainedModel, PreTrainedTokenizer

from .logits_processors import (
    NumberLogitsProcessor,
    LiteralLogitsProcessor,
    NumberStoppingCriteria,
    StringStoppingCriteria,
)


class ConstrainedOutput(BaseModel):
    value: Any
    array_closed: bool = False


def check_array_closed(text: str):
    text = text.strip()
    if text.endswith("]"):
        return True
    if text.endswith("],"):
        return True
    return False


def check_control_flags(text: str):
    return check_array_closed(text)


def generate_number(
    type_: Union[Type[float], Type[int]],
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    prompt: str,
    max_number_tokens: int,
    temperature: Union[float, None] = None,
) -> ConstrainedOutput:
    """
    generate number

    Args:
        type_: int or float
        prompt: direct input for model
        max_number_tokens: control parameter for generation
        temperature: control parameter for generation
    """
    input_tokens = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    number_logits_processor = NumberLogitsProcessor(tokenizer)

    output = model.generate(
        input_tokens,
        max_new_tokens=max_number_tokens,
        num_return_sequences=1,
        logits_processor=[number_logits_processor],
        stopping_criteria=[NumberStoppingCriteria(tokenizer, len(input_tokens[0]))],
        temperature=temperature or temperature,
        pad_token_id=tokenizer.eos_token_id,
    )
    output = output[0][len(input_tokens[0]) :]
    output = tokenizer.decode(output, skip_special_tokens=True)
    # remove possible trailing point
    output = output.strip().rstrip(".")
    # detect if model thinks there is an array to finish
    array_closed = check_control_flags(output)
    # remove possible array close symbols
    if array_closed:
        output = output.rstrip(",").rstrip("]")

    logging.debug("[generate_number] " + output)
    return ConstrainedOutput(value=type_(float(output)), array_closed=array_closed)


def generate_string(
    prompt: str,
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    max_string_token_length: int,
    temperature: Union[float, None] = None,
) -> ConstrainedOutput:
    """
    generate string

    Args:
        prompt: direct input for model
        max_string_token_length: control parameter for generation
        temperature: control parameter for generation
    """
    input_tokens = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

    output = model.generate(
        input_tokens,
        max_new_tokens=max_string_token_length,
        num_return_sequences=1,
        temperature=temperature,
        stopping_criteria=[StringStoppingCriteria(tokenizer, len(input_tokens[0]))],
        pad_token_id=tokenizer.eos_token_id,
    )
    output = output[0][len(input_tokens[0]) :]
    output = tokenizer.decode(output, skip_special_tokens=True)

    # detect if model thinks there is an array to finish
    array_closed = check_control_flags(output)
    # cut from first " symbol
    if output.count('"') > 0:
        output = output.split('"')[0].strip()

    logging.debug("[generate_string] " + output)
    return ConstrainedOutput(value=output, array_closed=array_closed)


def generate_literal(
    prompt: str,
    values: List[str],
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    temperature: Union[float, None] = None,
) -> ConstrainedOutput:
    """
    generate result from given candidate values

    Args:
        prompt: direct input for model
        values: candidate values
        temperature: control parameter for generation
    """
    input_tokens = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    literal_logits_processor = LiteralLogitsProcessor(literals=values, tokenizer=tokenizer)

    output = model.generate(
        input_tokens,
        max_new_tokens=2048,
        num_return_sequences=1,
        temperature=temperature,
        logits_processor=[literal_logits_processor],
        stopping_criteria=[StringStoppingCriteria(tokenizer, len(input_tokens[0]))],
        pad_token_id=tokenizer.eos_token_id,
    )
    output = output[0][len(input_tokens[0]) :]
    output = tokenizer.decode(output, skip_special_tokens=True)

    # detect if model thinks there is an array to finish
    array_closed = check_control_flags(output)
    # cut from first " symbol
    if output.count('"') > 0:
        output = output.split('"')[0].strip()

    logging.debug("[generate_literal] " + str(output))
    return ConstrainedOutput(value=output, array_closed=array_closed)
