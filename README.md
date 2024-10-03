# HARPNET - High-Resolution Attention Res-UNet Precipitation Network

## The problem
Many weather model grids are too coarse to resolve fine terrain influences, such as topography or even land/water borders, which has profound implications for the precipitation values within that grid. While a perfect model at that resolution would predict the correct mean for all the points within that grid, the actual values at higher resolution would vary from that mean significantly. The goal of downscaling is to accurately go from coarse, low-resolution grids to high-resolution grids that can accurately resolve these terrain influences.

Many techniques have been attempted, with varying degrees of success. One approach to downscaling involves calculating "downscaling ratios," which are static multipliers applied to coarse precipitation inputs that have been interpolated to the target high resolution. These ratios adjust the interpolated coarse values to more accurately reflect the fine-scale influences of the terrain, improving the precision of the forecast. These ratios are often computed by simply coarsening high-resolution precipitation climatology datasets such as PRISM or CHELSA to the target input resolution – say 0.25 degrees – before interpolating the coarsened climatology back to the target resolution. After this step, the high-resolution target grid values are divided by the interpolated coarse grid values, giving multipliers – or ratios – that are necessary to get from the coarsened grid values to the high-resolution target grid values. This technique is easy and computationally efficient, and for the most part does mostly improve error. However, since the ratios are built off of climatology data, they perform best for "average" events and perform poorly for fringe/rare events. 

For example: imagine a mountainous location with plains to the south and large mountains to the north. If this location recieves most of its precipitation from southerly wind events due to favorable orographic forcing (no obstructing terrain), then the downscaling ratios will perform well and will accurately downscale the coarse precipitation inputs to a high-resolution grid. However, during an event with winds out of the north, applying these same multipliers will overdo the high-resolution precipitation, since the ratios are static and have no concept that some wind directions are more favorable for small-scale enhancement of precipitation than others.

Regardless of which method is used, the problem with any static downscaling method is that it's not dynamic, it can't respond differently to unique events and downscale the precipitation to a high-resolution grid accordingly. The solution to this is dynamic downscaling, which downscales differently depending on the event setup. One method of dynamic downscaling is running a physical mesoscale model, such as WRF, initialized from the coarse grid initial & boundary conditions. Of course, there are errors in the equations and parameterizations used in these models themselves, but generally speaking, dynamically downscaling in this manner is far more accurate than any static downscaling method. The main drawback to this method is the required computation, as it requires solving millions of equations at each timestep, resulting in significant computational overhead. Additionally, dynamically downscaling using physical models is non-modular, meaning to downscale one variable, you must predict all other variables at the target resolution, even if they're not needed.

So, machine learning (ML) presents a unique opportunity for dynamic downscaling while retaining computational efficiency. ML models can accurately downscale depending on the synoptic setup, despite being far more computationally efficient than using physical models for downscaling. Additionally, ML models are modular, meaning you can predict Y number of outputs from X number of inputs, without requiring unnecessary variables to be involved in any manner along the way.

## Model Architecture
![alt text](https://github.com/clamalo/harpnet/blob/master/figures/harpnet.png?raw=true)
HARPNET uses an attention-gated residual convolution UNet architecture to downscale precipitation. The advantages of this architecture include efficiency and the fact that all the data is processed together, which is important for retaining spatial continuity throughout the downscaling process. This architecture was decided on after several considerations:

- **UNet:** The UNet's architecture consists of an encoder that captures context through down-sampling and a decoder that enables precise localization through up-sampling. This structure is ideal for converting low-resolution inputs into high-resolution outputs while preserving important spatial information; it gives the network the best of both worlds, rich feature extraction while retaining crucial spatial context. The UNet architecture includes skip connections that directly link corresponding layers in the encoder and decoder. These connections help retain high-resolution features lost during down-sampling via pooling layers and improve the accuracy of the output.
  
- **Custom ResConvBlocks rather than ConvBlocks:** Residual connections help mitigate the vanishing gradient problem, which allows for training deeper networks. By adding the input of a layer to its output, residual connections ensure that the network can learn the identity function more easily. This helps in retaining important features and gradients during backpropagation. The skip connections in Res-UNet allow the network to learn and propagate low-level features directly to deeper layers. This is particularly useful for tasks like precipitation downscaling where preserving fine details is crucial. Residual connections improve the flow of gradients through the network during training. This leads to faster convergence and can help avoid issues related to poor initialization.
  
- **Attention gates in the decoder steps of the network:** Attention gates on the spatial tensors from across the network before the skip connection help the network focus on the most relevant parts of the input features. They allow the model to dynamically weigh the importance of different spatial locations, enhancing the ability to capture important structures and patterns in the data. By selectively highlighting relevant features and suppressing irrelevant ones, attention mechanisms reduce noise and irrelevant information. This results in cleaner and more accurate output predictions, especially in complex tasks like precipitation downscaling. The decoder reconstructs the high-resolution output using features from the encoder. Attention gates can improve the fusion of these features by assigning higher weights to the more informative encoder features, leading to better feature integration and more precise reconstructions.

- **Dropout Layers:** Dropout layers were added in the decoder steps with decreasing dropout rates with increasing spatial complexity up the UNet, from 0.5 in the bottom of the model to 0.1 in the final decoder step before the output convolution. Dropout layers are incorporated in the decoder steps of the HARPNET architecture to enhance the model's generalization ability and prevent overfitting. No dropout layers were included in the encoder steps to ensure that no spatial information or extracted features are lost.

## Modularity
As mentioned before, downscaling using physical models is not modular – you must compute and downscale all variables together, even if they're not wanted as output variables. However, this is not true when downscaling with ML models; you can process variables modularly where the model only learns relationships and dependencies between specifically selected inputs and outputs. One of the special things about the HARPNET architecture is that increasing the number of input and output variables does not actually significantly increase the computational complexity; in fact, increasing the number of input/output variables from 1 to 64 only increases the numbers of parameters by 0.1%. This is because increasing the number of inputs only increases the complexity of the first layer of the model; it doesn't affect any other "tier" of the UNet, and vice versa for the outputs. This holds true until 65 input or output variables is reached, as the first layer would then be shrinking the data complexity in the first layer, leading to a loss of feature information. If this limit were reached, you would simply remove the first tier of the UNet and use the 128-channel tier as the first layer. This means that the variable complexity limit for HARPNET is largely dictated by storage size for the training data and memory capacity, not by computing capacity, leading to faster training times and more stable convergence. 

This fact opens the door up for a variety of future work, including seeing if adding more input/output variables improves the accuracy of precipitation predictions. Additionally, this opens up the door for some sort of full end-to-end machine learning mesoscale model emulator with an entire suite of traditional forecast variables.

## Data
HARPNET was trained using CONUS404, a 4km reanalysis dataset over CONUS prepared by the NOAA. CONUS404 was created by dynamically downscaling hourly native ERA5 data from ~25km to ~4km using WRF.

The input data was constructed by interpolating the 4km 3-hourly summed CONUS404 data to a 0.25-degree reference grid, emulating the input conditions of a global model like the GFS or ECMWF (or their ensemble counterparts). The target data was native 4km CONUS404 data remapped to a WGS84 coordinate reference system and coarsened to 0.0625 degree resolution, exactly 1/4th the grid spacing of a 0.25 degree model so that the edges of the coarse and fine grids can be perfectly aligned. 

Hourly precipitation data from 0z October 1, 1979 through 23z September 30, 2022 was used in the creation of HARPNET. This hourly data was summed into 3-hourly chunks, since HARPNET predicts 3-hourly precipitation. These 3-hourly chunks were from 0-3z, 3-6z, 6-9z, etc. The training and test sets were generated with data from 0z October 1, 1979 to 23z September 30, 2021 using a random 20% train/test split. Consistent random seeding was employed in numpy, pytorch, and Python's random package to ensure consistent train/test splits across tiles and across different runs.

The validation set was constructed using data from 0z October 1, 2021 through 23z September 30, 2022 in order to have sequential unseen data to use in tests that are more temporally-dependent, such as station observations and complete storm cycles.

HARPNET is trained to predict 64x64 target grid tiles. Training smaller til models rather than a single larger models allows for increased computational efficiency and more modularity; these tiles can be stitched together to downscale large areas at a time while remaining computationally efficient. The input grids were cropped to give exactly 0.25 degrees of buffer around the target grids to ensure no input information was lost around the edges of each domain; this was done thanks to remapping the CONUS404 data so the edges perfectly line up with the 0.25 degree input grids.

## Training
HARPNET was trained on a 2023 M2 Max MacBook Pro. The following hyperparameters were used for each til:
- **Optimizer:** Adam
- **Learning rate:** 1e-4
- **Batch size:** 32
- **Epochs:** 20

After each epoch, the model and optimizer states were saved as checkpoints.

## HARPNET Ensemble
HARPNET was trained to be able to have an ensemble component, as well (HARPNET-E). By treating each epoch checkpoint state dictionary as a different member, HARPNET-E accounts for uncertainty in the downscaling process and can create an ensemble of solutions from a single deterministic input. Some members are more skillful than others, but the ensemble mean has proven to be more skillful than any given individual member across a large enough sample size of events ***(need to test)***.

FUTURE WORK: train "true" ensemble by training multiple members in parallel:
- Perturbing the CONUS404 data with noise, then re-coarsening, training unique members on noised data
- Bagging: train models on subsets of the data and with different initial seeds
   - Aggregate at the parameter-level by combining weights rather than aggregating the outputs...
 
## Results
Our tests show that HARPNET improves high-resolution precipitation forecasts when compared to traditional upsampling methods such as bilinear interpolation and statistical downscaling.

The model seems to pick up on atmospheric features that are not explicitly fed to the model; it downscales precipitation differently depending on wind direction, storm/synoptic setup, etc.
- Run through a few test-cases displaying synoptic awareness (operational or in test data?)

PLOTS:
1.) Data workflow
2.) Low-res vs CONUS404 climatology by month (side by side 4 panel plots)
3.) Train/test loss curves
4.) Model high resolution MSE/RMSE by grid point compared to CONUS404
5.) Bilinear interpolation high resolution MSE/RMSE by grid point compared to CONUS404
6.) Statistically downscaled high resolution MSE/RMSE by grid point compared to CONUS404
NEW CHARTS: 
- Model MSE reduction compared to bilinear interpolation
- Model MSE reduction compared to statistical downscaling
7.) Dynamic ratios depending on wind (WY2021-2022)
8.) Convection vs orographic ratios (WY2021-2022)
9.) Station data (orographic event in WY2021-2022)
10.) Station-validated metrics table

## Future Work
- Train ECMWF/GFS specific model weights (CONUS404 for ECMWF, HRRR for GFS?)
- Train with other variable inputs (Pressure levels of U/V, Temp, GH, as well as PWAT, CAPE, TEMP)
    - Evaluate most important input features by feeding noise for certain inputs and seeing the impact on accuracy
- Physical preservation of grid precipitation totals
- Train different models for each season
- Employing generative adversarial networks to generate even more realistic downscaled forecasts
- Additional batches of trained tiles staggered by 50% latitude/longitude, outputs are blended among the overlapping tiles to almost eliminate edge-to-edge discontinuity
- Custom loss to encourage even better tile-to-tile continuity
- Train using ERA5 precip input... maybe ERA5 -(WRF)> CONUS404 -(coarsen)> != ERA5, so much so that the model only learns how to predict high-res using coarsened high-res?
- Native models trained for each model resolution (ICON, GDPS, Sflux, etc)
- Use entire training set... 3-hourly chunks every hour instead of every 3 hours
- Mixed training dataset to allow the model to predict hourly, 3-hourly, and 6-hourly data
- Upgrading from 0.0625 degree resolution to 0.03125 degree resolution
