
import torch


def generate(
    model,
    input_ids,
    max_new_tokens=30
):

    model.eval()

    generated_ids = input_ids.clone()
    past_key_values = None

    with torch.no_grad():

        for step in range(max_new_tokens):

            if past_key_values is None:
                # First iteration: process the complete prompt
                current_input = generated_ids
            else:
                # Later iterations: process only the newest token
                current_input = generated_ids[:, -1:]

            logits, past_key_values = model(
                current_input,
                past_key_values=past_key_values
            )

            # Select the next token
            next_token_logits = logits[:, -1, :]

            next_token = torch.argmax(
                next_token_logits,
                dim=-1,
                keepdim=True
            )

            generated_ids = torch.cat(
                [generated_ids, next_token],
                dim=1
            )

    return generated_ids
