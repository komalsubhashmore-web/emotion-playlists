# From Emotions to Playlists: Facial Emotion Recognition for Music Recommendation

The emotion-recognition core of a music recommender that reads a listener's facial expression and uses it to choose music. This repo trains a 7-class facial emotion classifier on three merged public datasets and exports it to ONNX for fast, on-device inference.

Advanced AI course project, M.S. in Artificial Intelligence, Dakota State University. Team: Komal Subhash More and Piyush Mishra.

## Why

Most recommenders rely on listening history and ignore how the listener feels right now. Our literature review found two gaps: systems rarely model emotion continuously (valence and arousal) instead of as fixed labels, and few use sequential or reinforcement learning to steer mood over a playlist. This project builds the first piece: reliable emotion recognition that can run locally, so face images never leave the user's device.

## Data

Three public facial-expression datasets, cleaned and merged into one label scheme:

| Dataset | Images |
|---|---|
| AffectNet (YOLO-format subset) | 25,262 |
| RAF-DB | 15,339 |
| FERPlus | 68,843 |
| **Combined** | **109,444** |

Labels were unified to 7 classes: angry, disgust, fear, happy, sad, surprise, neutral. FERPlus "contempt" and the extra AffectNet class were merged into disgust. Data was split 85/15 with stratification by emotion (93,027 train, 16,417 validation).

> Datasets are not included in this repo. Download each one from its official source under its own license.

## Model

- **Backbone:** EfficientNet-B2 pretrained on ImageNet (via `timm`), with a new 7-class head
- **Augmentation:** random resized crop, horizontal flip, color jitter, ImageNet normalization
- **Training:** AdamW (lr 1e-4), cross-entropy loss, mixed-precision (AMP) on GPU, 12 epochs, best checkpoint saved by validation accuracy

## Results

| | Accuracy |
|---|---|
| Best validation accuracy | **86.2%** (epoch 11) |
| Final train accuracy | 85.6% |

Validation accuracy tracked training accuracy closely, so the model was not badly overfitting.

## Deployment: ONNX export

- The trained classifier is exported to ONNX (`opset 17`, dynamic batch size) for use with ONNX Runtime.
- A second notebook wraps the pretrained [EmoNet](https://github.com/face-analysis/emonet) model to output continuous **valence and arousal** and exports it to ONNX (`opset 18`). EmoNet is third-party work; see its repository for its license and weights.

## Notebooks

| Notebook | What it does |
|---|---|
| `Categorical_FER_Model.ipynb` | Merges datasets, trains EfficientNet-B2, exports to ONNX |
| `EmoNet_ONNX_Export.ipynb` | Wraps EmoNet for valence-arousal output and exports to ONNX |

Both were built in Google Colab. Update the Google Drive paths to your own data location before running.

## Next steps

- Map predicted emotion and valence-arousal to audio features (tempo, energy) to build playlists
- Use reinforcement learning to move a listener toward a target mood over a sequence of songs
- Test fairness across demographics, lighting and occlusions such as glasses or masks

## Literature review

`From_Emotions_to_Playlists.pdf` surveys facial emotion recognition, categorical vs. continuous emotion models, multimodal recognition, and privacy for emotion-aware music systems.
