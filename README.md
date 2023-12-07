# RecogNYCe

Playing Geoguessr in NYC using deep learning superpowers.

The codebase consists of three notebooks:

- `scraping.ipynb`: Construction of the RecogNYCe dataset.
- `training.ipynb`: Training of the neural networks.
- `results.ipynb`: Model evaluation, webapp log processing and result visualization.

The web application is also contained within this repository and can be found in the `./webapp` directory.

In order to keep the repository lightweight, both the dataset and the model weights are distributed through Google Drive:

- [Dataset](https://drive.google.com/file/d/1uf4-sjmTlpDPRc4nTaanUCLmzOH8y1C9/view?usp=sharing)
- [Model Weights](https://drive.google.com/file/d/1h0h4MQMAgINH07Pp93Ap7VjfJB02hGxw/view?usp=drive_link)

The complete folder structure for this project looks like.

```
.
├── data
│   └── all
│       └── ...
├── docker-compose.yml
├── figures
├── models
│   ├── cnn_benchmark.pth
│   ├── priors.npy
│   ├── regnet_5_alone.pth
│   ├── resnet18_20+5_cont.pth
│   ├── resnet18_places_5_-2_alone.pth
│   ├── resnet50_5_alone.pth
│   └── results
│       ├── accuracy.npy
│       ├── y_hat.npy
│       └── y.npy
├── nginx.conf
├── README.md
├── results.ipynb
├── scraping.ipynb
├── training.ipynb
├── utils
│   ├── geojson_utils.py
│   ├── gmaps_utils.py
│   ├── mapillary_utils.py
│   ├── model_utils.py
│   └── plot_utils.py
└── webapp
    └── ...
```
