import fastembed
print(dir(fastembed))
try:
    from fastembed.image import ImageEmbedding
    print("Found in fastembed.image")
except ImportError:
    pass

try:
    from fastembed.vision import ImageEmbedding
    print("Found in fastembed.vision")
except ImportError:
    pass
