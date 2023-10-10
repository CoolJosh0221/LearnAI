#!/bin/bash

# Download the dogs-vs-cats dataset
kaggle competitions download -c dogs-vs-cats

# Unzip the dogs-vs-cats.zip file
unzip dogs-vs-cats.zip

# Enter the unzipped folder
cd dogs-vs-cats

# Unzip the train.zip file
unzip train.zip

# Delete the train.zip file
rm train.zip

# Return to the parent directory
cd ..

# Delete the dogs-vs-cats.zip file
rm ./dogs-vs-cats.zip
rm ./sampleSubmission.csv
rm ./test1.zip