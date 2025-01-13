
# CRP: Cluster, Reduce, Pass - A Novel GNN Architecture for Long-Range Graph Reasoning

Graph Neural Networks (GNNs) have emerged as a powerful framework for learning on graph-structured data, achieving remarkable success across domains like molecular property prediction and recommender systems. However, GNNs face challenges such as oversquashing and oversmoothing, which hinder their scalability and effectiveness in large graphs with long-range dependencies.  

To address these issues, we introduce **Cluster, Reduce, Pass (CRP)**, a novel architecture that builds on existing clustering and graph coarsening techniques. CRP leverages clustering algorithms (e.g., KMeans, GMM) to group similar nodes into clusters, reducing graph complexity while preserving long-range dependencies. By iteratively applying coarsening and message-passing, CRP mitigates oversquashing and oversmoothing, offering an efficient and scalable solution for large-scale graphs.  

This repository contains the implementation of CRP, tested on two datasets from the Long Range Graph Benchmark (LRGB). The results are compared with the CNA architecture and state-of-the-art (SOTA) models reported on LRGB datasets.  

---

![Add your project image or diagram here for better understanding]

---

## Creating the Conda Environment: CRP

Follow these steps to set up a Conda environment and install the required dependencies for this project:  

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/Cluster-Reduce-Pass/CRP.git
   cd CRP
   ```

2. **Create the Conda Environment**  
   ```bash
   conda create -n CRP python=3.9 -y
   conda activate CRP
   ```

3. **Install Dependencies**  
   Use the provided `requirements.txt` file to install the necessary libraries:  
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Installation**  
   To ensure all dependencies are correctly installed, run:  
   ```bash
   python -c "import torch; import torch_geometric; print('Environment setup successful!')"
   ```

5. **Run the Code**  
   Follow the instructions in the repository to run experiments and test the CRP architecture on LRGB datasets.  

---

Feel free to let me know if you'd like to adjust the text or add more details!
