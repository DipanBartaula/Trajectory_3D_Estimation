import torch
from model.text.hf_embedder import TextFeatureExtractor

def test_embedder():
    print("Testing TextFeatureExtractor...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        extractor = TextFeatureExtractor(device=device)
        extractor = extractor.to(torch.float16)
        
        caption = ["a sleek modern chair with wooden legs"]
        print(f"Input Caption: {caption}")
        
        with torch.no_grad():
            t5_feat, clip_feat = extractor(caption)
            
        print(f"T5 Features Shape: {t5_feat.shape}")
        print(f"CLIP Features Shape: {clip_feat.shape}")
        
        assert t5_feat.is_floating_point()
        assert clip_feat.is_floating_point()
        
        print("Embedder Test Passed!")
    except Exception as e:
        print(f"Embedder Test Failed: {e}")

if __name__ == "__main__":
    test_embedder()
