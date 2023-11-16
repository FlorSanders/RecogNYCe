# RecogNYCe

Playing Geoguessr in NYC using deep learning superpowers.

## Data

### Data acquisition strategy

1. Obtain latitude/longitude polygon descriptions of the borders of each borough/neighborhood in NYC
2. Sample random latitude/longitude coordinates within each polygon.
3. For each sampled sampled coordinate, scrape images from street-view-like service
4. Save both the image and the sampled coordinate for the image sample.
5. Manually go through the dataset and remove poor quality samples.

Result: An image dataset of street-level pictures for each of the boroughs / neighborhoods in NYC

### Data sources

- NYC location data
  - [Boroughs geojson](./data/location/nyc_boroughs.geojson) - [source](https://github.com/codeforgermany/click_that_hood/blob/main/public/data/new-york-city-boroughs.geojson)
  - [Neighborhoods geojson](./data/location/nyc_neighborhoods.geojson) - [source](https://github.com/veltman/snd3/blob/master/data/nyc-neighborhoods.geo.json)
- NYC street-level image data
  - [Mapillary](https://www.mapillary.com/) - [Python API](https://pypi.org/project/mapillary/)
  - [Google Street View](https://www.google.com/streetview/) - [Python API](https://pypi.org/project/google-streetview/)
- Own pictures?

## Models

<p style="color: red;">TODO: literature research on image geolocation.</p>

High-level idea, two possible approaches:

1. Regression:
   - regress on the latitude / longitude of the coordinates, training on the MSE between predicition & label
   - classify afterwards based on the borough in which the predicted
2. Classification:
   - classify immediately on the known class list of boroughs / neighborhoods

We could try to develop both models and compare their performance.

The performance for the classification problem should be compared to:

- [ ] Random choices
- [ ] Prior distribution choices
- [ ] Naive bayes method
- [ ] Simple LeNet architecture

Some nice visualizations would be

- [ ] Confusion matrix
- [ ] Scatter map with wrong/right points
- [ ] Precision vs recall plot
- [ ] Some examples of well-classified images & wrongly classified images
- [ ] Where our models score in the distribution of users
- [ ] t-SNE visualization of our image data / of the features by our learned network

## Outcome

- New dataset for street-level pictures for the boroughs / neigborhoods in NYC
- Model(s) trained for classification / geolocation based on these images
- (Possibly) minigame that allows you to play NYC Geoguessr against the AI
  - ## (Possibly) comparison of human and AI performance in this specific case based on logs
