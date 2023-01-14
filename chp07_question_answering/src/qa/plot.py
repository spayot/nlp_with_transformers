import datasets
import matplotlib.pyplot as plt
import numpy as np
import transformers as tfm


def plot_logits_as_barchart(
    inputs: tfm.tokenization_utils_base.BatchEncoding,
    outputs: tfm.modeling_outputs.QuestionAnsweringModelOutput,
    tokenizer: tfm.PreTrainedTokenizerBase,
) -> None:
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    s_scores = start_logits.detach().numpy().flatten()
    e_scores = end_logits.detach().numpy().flatten()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    token_ids = range(len(tokens))

    fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
    colors = ["C0" if s != np.max(s_scores) else "C1" for s in s_scores]
    ax1.bar(x=token_ids, height=s_scores, color=colors)
    ax1.set_ylabel("Start Scores")
    colors = ["C0" if s != np.max(e_scores) else "C1" for s in e_scores]
    ax2.bar(x=token_ids, height=e_scores, color=colors)
    ax2.set_ylabel("End Scores")
    plt.xticks(token_ids, tokens, rotation="vertical")
    plt.show()
