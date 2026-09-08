# SpheroQuaNNt

SpheroQuaNNt is a Shiny application for automated spheroid segmentation and volume estimation from microscopy images using a U-Net convolutional neural network.

## Running SpheroQuaNNt locally in R

To run SpheroQuaNNt locally, you will need the application file and the trained U-Net model.

### 1. Download the application

Download the `app.R` file from this GitHub repository.

### 2. Download the trained U-Net model

The trained U-Net model used by SpheroQuaNNt is available on Figshare.

Download the model from:

https://figshare.com/articles/software/UNet_SpheroQuaNNt_/33471091?file=68318365


### 3. Create a folder

Create a new folder on your computer and place the following two files in the same folder:

- `app.R`
- `unet_11march.h5`

### 4. Run the application in R

Open RStudio and set the newly created folder as your working directory.

Open `app.R` and run the application.

Alternatively, in RStudio, open `app.R` and click **Run App**.

The SpheroQuaNNt application will open in your default web browser.

### 5. Using the application

Upload your microscopy images using **Upload Microscopy Images**.

Click **Run Segmentation**.

The application will generate:

- Predicted spheroid segmentation overlays
- Minimum diameter (Dmin)
- Maximum diameter (Dmax)
- Estimated spheroid volume

Results can be downloaded as a CSV file, and predicted masks and overlays can be downloaded as a ZIP file.
