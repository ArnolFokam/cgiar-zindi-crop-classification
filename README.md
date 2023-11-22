# Get Started with ResNet50 baseline

1. Create a pyton environment and install the [requirements.txt](/requirements.txt)

2. Download the data with the following command

```bash
python download_data.py && unzip data/images.zip -d data && rm data/images.zip
```

3. Download the `Train.csv`, `Test.csv` and `SampleSubmission.csv` from [here](https://zindi.africa/competitions/cgiar-crop-damage-classification-challenge/data) and put them in the data folder.

4. Run the following command to train the model and get the results

```bash
python main.py
```

Note: you can modify the import in the [`main.py`](main.py) file to use you run training code.