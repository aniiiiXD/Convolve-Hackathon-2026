from fastembed import SparseTextEmbedding

try:
    print(SparseTextEmbedding.list_supported_models())
except Exception as e:
    print(e)
