# CNA Comparison

This folder contains an implementation to benchmark the CNA module. This requires the installation of our adapdted [CNA package](https://github.com/yzimmermann/CNA_Custom/tree/main).

## Configuration

You can modify several parameters in `CNA_comparison/LRGB_CNA.py` to customize the benchmarking. Most important are:

- **`dataset_name`**: Specifies the dataset to use. You can use `"Peptides-struct"` or `"Peptides-func"`.
  
- **`hidden_features`**: The dimensionality of hidden node embeddings.

- **`num_layer`**: The number of layers in the model.

- **`layer_type`**: The type of GNN layer used.

- **`activation`**: The activation function used in the model. You can e.g. use `torch.nn.GELU()` or switch to the CNA-specific activation by uncommenting the `RationalOnCluster` setup.


## Running the Script

After making the desired modifications, run the script with the following command:
```bash
python CNA_comparison/LRGB_CNA.py

