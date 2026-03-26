# Python inference

After training (`python train.py`), run predictions on images in the `images/` directory:

```bash
source venv/bin/activate
python predict.py
```

This loads `cats_dogs_model.keras`, classifies each image, and displays the result with matplotlib.
