import json
import os
import torch
import gradio as gr
import time

from mosestokenizer import MosesTokenizer
from mosestokenizer import MosesSentenceSplitter
from diffmask.models.nli_classification_diffmask import (
    BertSentimentClassificationSSTDiffMask,
)

from typing import List, Tuple


class WordImportanceScorer:
    """
    Class for scoring the importance of words in a given text using a pre-trained BERT model.

    Args:
        model_path (str): The path to the pre-trained BERT model.
        language (str, optional): The language of the text. Defaults to "en".
        device (str, optional): The device to run the model on. Defaults to "cpu".
    """

    def __init__(self, model_path, language="en", device="cpu"):
        """
        Initializes the WordImportanceScorer class.

        Args:
            model_path (str): The path to the pre-trained BERT model.
            language (str, optional): The language of the text. Defaults to "en".
            device (str, optional): The device to run the model on. Defaults to "cpu".
        """
        self.device = device
        self.model = BertSentimentClassificationSSTDiffMask.load_from_checkpoint(
            model_path
        ).to(device)
        self.model.freeze()
        
        self.mt = MosesTokenizer(lang=language)
        self.mss = MosesSentenceSplitter(lang=language)

    def tokenize(self, sentences: List[str]) -> List[str]:
        """
        Tokenizes a list of sentences.

        Args:
            sentences (List[str]): The list of sentences to tokenize.

        Returns:
            List[str]: The tokenized sentences.
        """
        return [" ".join(self.mt(sentence)) for sentence in sentences]

    def split_sentences(self, text: str, split: str) -> List[str]:
        """
        Splits a text into sentences.

        Args:
            text (str): The text to split into sentences.
            split (str): The method of splitting. Can be "Line breaks" or "Moses".

        Returns:
            List[str]: The list of sentences.
        """
        if split == "Line breaks":
            return text.split("\n")
        elif split == "Moses":
            return self.mss([text])
        else:
            return [text.replace("\n", " ")]

    def get_attributions(self, encoded_lines, encoded_lines_lengths):
        """
        Calculates the word attributions for the encoded lines.

        Args:
            encoded_lines: The encoded lines.
            encoded_lines_lengths: The lengths of the encoded lines.

        Returns:
            The word attributions.
        """
        enc = torch.tensor(encoded_lines, device=self.device, dtype=torch.long)
        len = torch.tensor(encoded_lines_lengths, device=self.device, dtype=torch.long)

        encoded_lines, encoded_lines_lengths = [], []

        return (
            self.model.forward_explainer(enc, len, enc, len, attribution=True)
            .exp()
            .cpu()
        )

    def generate_masks(
        self, sentences, split, tokenize, batch_size=32, seed=0
    ) -> Tuple[List[str], List[List[float]]]:
        """
        Generates masks for the given sentences.

        Args:
            sentences: The sentences to generate masks for.
            split: The method of splitting. Can be "Line breaks" or "Moses".
            tokenize: Whether to tokenize the sentences.
            batch_size: The batch size for processing the sentences. Defaults to 32.
            seed: The seed for random number generation. Defaults to 0.

        Returns:
            Tuple[List[str], List[List[float]]]: The lines and their corresponding masks.
        """
        torch.manual_seed(seed)

        lines, lines_lengths, encoded_lines, encoded_lines_lengths, attributions = (
            [],
            [],
            [],
            [],
            [],
        )

        sentences = self.split_sentences(sentences, split)

        if tokenize:
            sentences = self.tokenize(sentences)

        for i, line in enumerate(sentences):
            lines.append(line.strip())

            # converting tokens to numbers based on the dictionary
            encoded = self.model.tokenizer.encode_line(line, add_if_not_exist=False)[
                :128
            ].to(self.device)

            lines_lengths.append(encoded.numel())
            encoded_lines_lengths.append(encoded.numel())

            # aligning all the inputs to 128 tokens
            tokens = torch.cat([encoded, torch.ones(128 - len(encoded))]).tolist()
            encoded_lines.append(tokens)

            if (i + 1) % batch_size == 0:
                attributions += self.get_attributions(
                    encoded_lines, encoded_lines_lengths
                )
                encoded_lines, encoded_lines_lengths = [], []

        attributions += self.get_attributions(encoded_lines, encoded_lines_lengths)
        encoded_lines, encoded_lines_lengths = [], []

        values = [
            [v.tolist() for v in attr[: length - 1]]
            for length, attr in zip(lines_lengths, attributions)
        ]

        return lines, values

    def aggregate_masks(self, masks, mode="last", scale=False):
        """
        Aggregates the masks based on the specified mode.

        Args:
            masks: The masks to aggregate.
            mode: The aggregation mode. Can be "last" or "average". Defaults to "last".
            scale: Whether to scale the aggregated masks. Defaults to False.

        Returns:
            The aggregated masks.
        """
        result = []
        for mask in masks:
            mask = torch.tensor(mask)
            for _ in range(mask.size(0)):
                to_plot = mask / mask.abs().max(0, keepdim=True).values

                if mode == "last":
                    min_val = to_plot[..., -1].min()
                    last_scaled = (to_plot[..., -1] - min_val) / (
                        to_plot[..., -1] - min_val
                    ).max()

                    result.append(to_plot[..., -1] if not scale else last_scaled)
                    break
                elif mode == "average":
                    average = torch.mean(to_plot, 1)
                    average_min = average.min()
                    average_scaled = (average - average_min) / (
                        average - average_min
                    ).max()

                    result.append(average if not scale else average_scaled)
                    break
                else:
                    raise "Aggregating mode `{mode}` is not recognized. Use --mode {last,average} instead."
        return result


if __name__ == "__main__":
    scorer_nli = WordImportanceScorer(
        "./interpreters/nli/checkpoint4.ckpt", device="cpu"
    )
    scorer_pi = WordImportanceScorer("./interpreters/pi/checkpoint4.ckpt", device="cpu")

    active_tab = 0

    def set_active_tab(evt: gr.SelectData):
        global active_tab
        active_tab = evt.index

    with gr.Blocks() as demo:
        with gr.Row():
            gr.Markdown(
                """
                # Word Importance Scorer
                An interactive tool for assessing word importance using models trained for semantic tasks.
                Please find details in the paper [Assessing Word Importance Using Models Trained for Semantic Tasks](https://aclanthology.org/2023.findings-acl.563/)
                and in the [GitHub repository](https://github.com/j4vorsky/word-importance).
                """
            )

        with gr.Row():
            with gr.Tab("Type text") as text_tab:
                sentence_box = gr.Textbox(
                    lines=9, label="Sentences (max 128 tokens each)"
                )
            with gr.Tab("Upload text file") as file_tab:
                upload_box = gr.File(label="Text file", file_types=["txt"])

            text_tab.select(set_active_tab)
            file_tab.select(set_active_tab)

        with gr.Row():
            model_box = gr.Dropdown(
                choices=["Paraphrase Identification", "Natural Language Inference"],
                value="Paraphrase Identification",
                label="Underlying model",
            )
            tokenize_box = gr.Radio(
                ["Do not tokenize", "Tokenize"],
                value="Do not tokenize",
                type="index",
                label="Tokenization",
            )
            split_box = gr.Radio(
                ["No split", "Line breaks", "Moses"],
                value="No split",
                label="Sentence split",
            )
            mode_box = gr.Radio(
                ["Last layer", "Average all layers"], value="Last layer", label="Layer"
            )

        with gr.Row():
            scale_box = gr.Radio(
                ["Original", "Scale to [0,1] within one sentence"],
                value="Original",
                type="index",
                label="Importance scores",
            )
            precision_box = gr.Slider(
                minimum=1, maximum=10, value=2, step=1, label="Float precision"
            )
            columns_box = gr.Slider(
                minimum=1, maximum=30, value=15, step=1, label="Display columns"
            )

        submit_btn = gr.Button("Submit")

        output_dataframe = gr.Dataframe(visible=False)

        with gr.Row(visible=False) as output_files:
            time_mark = time.ctime().replace(" ", "-")
            output_tsv_file = gr.Button("Download .tsv", link=f"/file=tmp/output.tsv")
            output_json_file = gr.Button(
                "Download .json", link=f"/file=tmp/output.json"
            )

        def create_output(
            model,
            sentence,
            upload,
            tokenize,
            split,
            scale,
            mode,
            precision,
            columns_count,
        ):
            if upload is not None and active_tab == 1:
                with open(upload, "r", encoding="utf8") as file:
                    sentences = file.read()
            elif upload is None and active_tab == 1:
                sentences = ""
            else:
                sentences = sentence

            if sentences == "":
                raise gr.Error("Empty input. Please provide some text.")

            mode = mode.split()[0].lower()

            scorer = scorer_pi if model == "Paraphrase Identification" else scorer_nli

            sentences, scores = scorer.generate_masks(sentences, split, tokenize)
            importance_scores = scorer.aggregate_masks(scores, mode=mode, scale=scale)

            agg_sentences = [
                token for sentence in sentences for token in sentence.split()
            ]
            agg_scores = torch.cat(importance_scores).tolist()

            agg_scores = [
                ("{0:." + str(precision) + "f}").format(v) for v in agg_scores
            ]

            result = []
            for i in range(0, len(agg_scores), columns_count):
                result.append(agg_sentences[i : i + columns_count])
                result.append(agg_scores[i : i + columns_count])

            result[-2] += [""] * (columns_count - len(result[-2]))
            result[-1] += [""] * (columns_count - len(result[-1]))

            with open("./tmp/output.tsv", "w", encoding="utf8") as file:
                for i in range(len(sentences)):
                    file.write(str(sentences[i]))
                    file.write("\t")
                    file.write(
                        " ".join(
                            [
                                ("{0:." + str(precision) + "f}").format(x)
                                for x in importance_scores[i]
                            ]
                        )
                    )
                    file.write("\n")

            with open("./tmp/output.json", "w", encoding="utf8") as file:
                json_result = {
                    "sentences": [
                        {"sentence": s.split(), "scores": imp.tolist()}
                        for s, imp in zip(sentences, importance_scores)
                    ]
                }
                json_object = json.dumps(json_result, indent=4)
                file.write(json_object)

            return {
                output_dataframe: gr.Dataframe(result, visible=True),
                output_files: gr.Row(visible=True),
            }

        submit_btn.click(
            create_output,
            [
                model_box,
                sentence_box,
                upload_box,
                tokenize_box,
                split_box,
                scale_box,
                mode_box,
                precision_box,
                columns_box,
            ],
            [output_dataframe, output_files],
        )

    demo.launch(allowed_paths=["./tmp"], share=True)
