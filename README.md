# HARPNET - **H**igh-Resolution Attention Res-UNet Precipitation Network
### OpenSnow's Next-Generation Precipitation Downscaling System

## Model Architecture
HARPNET uses an attention-gated residual convolution UNet architecture to downscale precipitation. The advantages of this architecture include efficiency and the fact that all the data is processed together, which is important for retaining spatial continuity throughout the downscaling process. This architecture was decided on after several considerations:

- **UNet:** The UNet's architecture consists of an encoder that captures context through down-sampling and a decoder that enables precise localization through up-sampling. This structure is ideal for converting low-resolution inputs into high-resolution outputs while preserving important spatial information; it gives the network the best of both worlds, rich feature extraction while retaining crucial spatial context. The UNet architecture includes skip connections that directly link corresponding layers in the encoder and decoder. These connections help retain high-resolution features lost during down-sampling via pooling layers and improve the accuracy of the output.
  
- **Custom ResConvBlocks rather than ConvBlocks:** Residual connections help mitigate the vanishing gradient problem, which allows for training deeper networks. By adding the input of a layer to its output, residual connections ensure that the network can learn the identity function more easily. This helps in retaining important features and gradients during backpropagation. The skip connections in Res-UNet allow the network to learn and propagate low-level features directly to deeper layers. This is particularly useful for tasks like precipitation downscaling where preserving fine details is crucial. Residual connections improve the flow of gradients through the network during training. This leads to faster convergence and can help avoid issues related to poor initialization.
  
- **Attention gates in the decoder steps of the network:** Attention gates on the spatial tensors from across the network before the skip connection help the network focus on the most relevant parts of the input features. They allow the model to dynamically weigh the importance of different spatial locations, enhancing the ability to capture important structures and patterns in the data. By selectively highlighting relevant features and suppressing irrelevant ones, attention mechanisms can reduce noise and irrelevant information. This results in cleaner and more accurate output predictions, especially in complex tasks like precipitation downscaling. The decoder reconstructs the high-resolution output using features from the encoder. Attention gates can improve the fusion of these features by assigning higher weights to the more informative encoder features, leading to better feature integration and more precise reconstructions.

Dropout layers were added in the decoder steps with decreasing dropout rates with increasing spatial complexity up the UNet, from 0.5 in the bottom of the model to 0.1 in the final decoder step before the output convolution.

## Data
HARPNET was trained using CONUS404, a 4km reanalysis dataset over CONUS prepared by the NOAA. CONUS404 was created by dynamically downscaling hourly native ERA5 data from ~25km to ~4km using WRF.

The input data was constructed by interpolating the 4km 3-hourly summed CONUS404 data to a 0.25-degree reference grid, emulating the input conditions of a global model like the GFS or ECMWF (or their ensemble counterparts).

Hourly precipitation data from 0z October 1, 1979 through 23z September 30, 2022 was used in the creation of HARPNET. This hourly data was summed into 3-hourly chunks, since HARPNET predicts 3-hourly precipitation. These 3-hourly chunks were from 0-3z, 3-6z, 6-9z, etc.

The training set was comprised of data from 0z October 1, 1979 through 21z September 30, 2020. The test set was comprised of data from 0z October 1, 1979 through 21z September 30, 2022.

The target grids were coarsened from native 4km resolution to 8km to save on computing.

The target grids were cropped to a desired domain. The input grids were cropped to give a 0.5 degree buffer around the target grids to ensure no information is lost around the edges of the domain.

## Training
HARPNET was trained on a 2023 M2 MacBook Pro. The following hyperparameters were used:
- Optimizer: Adam
- Learning rate: 1e-3
- Batch size: 256
- Epochs: 50

After each epoch, the model and optimizer states were saved as checkpoints.

## HARPNET Ensemble
HARPNET was trained to be able to have an ensemble component, as well (HARPNET-E). By treating each epoch checkpoint state dict as a different member, HARPNET-E accounts for uncertainty in the downscaling process and can create an ensemble of solutions from a single deterministic input. Some members are more skillful than others, but the ensemble mean has proven to be more skillful than any given individual member.
