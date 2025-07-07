<img src="./contrail.png" alt="Header" width="400" />


# Google Research: Identify Contrails to Reduce Global Warming

This repository contains code and experiments for the [Kaggle Competition: Google Research - Identify Contrails to Reduce Global Warming](https://www.kaggle.com/competitions/google-research-identify-contrails-reduce-global-warming). The goal is to build an image segmentation model to detect contrails (airplane-induced cirrus clouds) in satellite imagery, enabling strategies to minimize radiative forcing.

---

## Data Description

* **train\_images/**: RGB satellite tiles (PNG) used for training.
* **train\_masks/**: Corresponding binary masks in run-length encoding for contrails.
* **test\_images/**: Unlabeled tiles for which you must predict contrail masks.
* **sample\_submission.csv**: CSV file with `image_id` and `encoded_pixels` columns.

Download and extract the `train` and `test` archives into the `data/` directory.

---

## Requirements

* Python 3.8+
* `numpy`
* `pandas`
* `opencv-python`
* `torch` & `torchvision` (or `tensorflow` if preferred)
* `segmentation-models-pytorch` (if using PyTorch)
* `albumentations`
* `tqdm`
* `matplotlib`

Install all dependencies via:

```bash
pip install -r requirements.txt
```

---

## Notebook Workflow (`unet-or-fcn2.ipynb`)

1. **Data Loading & Preprocessing**

   * Decode RLE masks into binary arrays.
   * Resize or tile images as needed.
2. **Augmentation Pipeline**

   * Random flips, rotations, brightness/contrast adjustments.
3. **Model Definitions**

   * **UNet** with a ResNet34 encoder.
   * **FCN-ResNet50** as an alternative baseline.
4. **Training & Validation**

   * Split training set into train/validation folds.
   * Define loss functions: Dice Loss + BCE.
   * Monitor IoU (Intersection over Union) and Dice score.
5. **Inference & Submission**

   * Apply model to `test_images/`.
   * Encode predicted masks back to RLE.
   * Generate `submission.csv`.

---

## Usage

1. Clone repository:

   ```bash
   ```

git clone \<repo\_url>
cd \<repo\_folder>

````

2. Install dependencies:
   ```bash
pip install -r requirements.txt
````

3. Prepare data:

   ```bash
   ```

mkdir -p data/train\_images data/train\_masks data/test\_images
tar -xvf train\_images.tar -C data/train\_images
tar -xvf train\_masks.tar -C data/train\_masks
tar -xvf test\_images.tar -C data/test\_images

````

4. Launch the notebook:
   ```bash
jupyter notebook unet-or-fcn2.ipynb
````

5. Run all cells to reproduce training, evaluation, and submission.

---

## Results

| Model         | Validation IoU | Validation Dice |
| ------------- | -------------: | --------------: |
| UNet-ResNet34 |           0.72 |            0.78 |
| FCN-ResNet50  |           0.68 |            0.74 |

The UNet-based architecture yielded better performance on held-out validation folds.

---

## Acknowledgments

* Google Research for hosting this competition.
* Kaggle community for insights and starter kernels.

---

## License

Distributed under the MIT License. See `LICENSE` for details.
