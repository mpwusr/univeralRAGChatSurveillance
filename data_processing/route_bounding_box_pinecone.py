from your_rag_module import embed_text, upsert_metadata  # customize these for your repo

for det in detections:
    caption = f"{det.class_name} with confidence {det.confidence:.2f}"
    embedding = embed_text(caption)  # or generate_image_embedding(crop) if using image patches

    metadata = {
        "label": det.class_name,
        "confidence": float(det.confidence),
        "coords": list(map(float, det.xyxy))
    }

    upsert_metadata(vector=embedding, metadata=metadata)
