🔗 **[loveless2001/qwen2.5-7b-glyph-sft](https://huggingface.co/loveless2001/qwen2.5-7b-glyph-sft)**

## Project Structure

*   `train/`: Scripts for fine-tuning the models (SFT).
*   `eval/`: Evaluation suites comparing Glyph, XML, and Natural language prompting strategies.
*   `data/`: specific datasets used for training and evaluation.
*   `checkpoints/`: Local storage for model checkpoints.
*   `experiments/`: Miscellaneous experiment scripts.

## Installation

1.  Clone the repository.
2.  Install the dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Inference

You can run a quick inference test using the provided script:

```bash
python infer_test.py
```

### Evaluation

To evaluate the model against different prompting strategies (Glyph vs. XML vs. Natural):

```bash
python eval/eval_structures.py --models loveless2001/qwen2.5-7b-glyph-sft --limit 20
```

### Training

Training scripts are located in the `train/` directory. For example, to run SFT:

```bash
python train/train_sft.py
```

## Glyph Dictionary

The project uses a set of alchemical glyphs to denote different stages of reasoning:

| Glyph | Name | Meaning/Usage |
| :--- | :--- | :--- |
| **🜞** | *Crux* | **Guideline**: Sets the intention or core rule for the problem. |
| **🜆** | *Flux* | **Plan**: Outlines the approach or strategy. |
| **🜂** | *Ignis* | **Step**: Execution of the reasoning steps or calculation. |
| **🜃** | *Terra* | **Takeaway**: The final answer or conclusion. |

## License

Apache 2.0
