from fastembed import TextEmbedding

try:
    print(TextEmbedding.list_supported_models())
except Exception as e:
    print(e)
