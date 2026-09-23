# The official code repository for the MVKcat-ET.

Extract offline protein language model embedding representations using the following Python libraries
[protT5 environment https://github.com/HannesStark/protein-localization.](https://pypi.org/project/bio-embeddings/)

*   **Dependency Library**:
    *   [bio-embeddings]
    *   [rdkit]
    *   [pytorch]
    *   [numpy]
    *   [pandas]
    *   [h5py]
    *   [scipy]
 
After having installed the bio-embeddings package, you can:

1. Use the pipeline like:

    ```bash
    bio_embeddings config.yml
    ```

2. Use the general purpose embedder objects via python, e.g.:

    ```python
    from bio_embeddings.embed import SeqVecEmbedder

    embedder = SeqVecEmbedder()

    embedding = embedder.embed("SEQVENCE")
    ```
After possessing the rdkit library, the physicochemical properties of molecules can be calculated.

Protein physicochemical characteristics can be calculated using Python's built-in libraries.
    
*   **Datasets Sources**:
    *   [Benchmark](https://github.com/LiLabTsinghua/GotEnzymes2) 
    *   [External Dataset](https://github.com/zchwang/CataPro)

The article has not been officially accepted yet, and the code is being gradually organized.
If you have any questions, please contact me directly.
 
