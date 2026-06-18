# src/main.py
import torch
import os
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.models.router import PacketRouter
from src.pipeline import MoEPipeline
from PIL import Image
from src.utils.channel import apply_awgn_channel, calculate_cosine_similarity, normalize_vector
from src.models.decoder import SemanticDecoder
from src.utils.benchmark import SemanticBenchmark



# =====================================================================
# 1. GENERAZIONE DI UN DATASET DI RETE SIMULATO
# =====================================================================
def generate_mock_network_data(num_samples=1000):
    """
    Genera dati di rete finti per l'addestramento del Router.
    Ogni campione ha 16 feature (metadati del pacchetto).
    """
    np.random.seed(42)
    X = []
    y = []
    
    for _ in range(num_samples):
        expert_type = np.random.choice([0, 1, 2]) # 0: Testo, 1: Audio, 2: Video
        
        # Creiamo un vettore di 16 feature basato sul tipo di esperto
        features = np.random.normal(0, 1, 16)
        
        if expert_type == 0:    # Pacchetti TESTO: Solitamente piccoli, porte standard HTTP (80/443)
            features[0] = np.random.uniform(0.1, 0.5)  # Dimensione pacchetto piccola
            features[1] = 80.0                          # Porta finta
        elif expert_type == 1:  # Pacchetti AUDIO: Dimensione media, flussi costanti
            features[0] = np.random.uniform(1.5, 3.0)  # Dimensione pacchetto media
            features[1] = 5004.0                        # Porta finta (RTP Audio)
        elif expert_type == 2:  # Pacchetti VIDEO: Molto grandi, picchi di banda elevati
            features[0] = np.random.uniform(5.0, 15.0) # Dimensione pacchetto grande
            features[1] = 1935.0                        # Porta finta (RTMP Video)
            
        X.append(features)
        y.append(expert_type)
        
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.long)

# =====================================================================
# 2. PIPELINE DI ADDESTRAMENTO DEL ROUTER
# =====================================================================
def train_router():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📥 Generazione dataset di rete simulato su {device}...")
    X_train, y_train = generate_mock_network_data(1200)
    
    # Inizializza il Router (16 feature in ingresso, 3 classi in uscita)
    router = PacketRouter(input_dim=16, num_classes=3).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(router.parameters(), lr=0.01)
    
    print("🏋️ Inizio addestramento del Router MLP...")
    router.train()
    
    X_train, y_train = X_train.to(device), y_train.to(device)
    
    for epoch in range(50): # 50 epoche bastano per dati così leggeri
        optimizer.zero_grad()
        outputs = router(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            # Calcola l'accuratezza al volo
            predictions = torch.argmax(outputs, dim=1)
            accuracy = (predictions == y_train).float().mean().item() * 100
            print(f"Epoch [{epoch+1}/50] | Loss: {loss.item():.4f} | Accuracy: {accuracy:.2f}%")
            
    # Salva i pesi del router addestrato nella cartella models/
    torch.save(router.state_dict(), "src/models/router_weights.pth")
    print("💾 Pesi del router salvati con successo in 'src/models/router_weights.pth'!")

# =====================================================================
# 3. AVVIO DELL'INFERENZA END-TO-END
# =====================================================================
if __name__ == "__main__":
    train_router()
    print("=" * 60)
    
    moe_pipeline = MoEPipeline(input_dim=16, router_weights_path="src/models/router_weights.pth")
    decoder_semantico = SemanticDecoder()
    
    # Inizializziamo il nostro profilatore hardware
    profiler = SemanticBenchmark()
    
    FOTO_REAL_PATH = "data/raw/foto_test.jpg"
    
    test_cases = [
        {
            "name": "STREAM TESTO (ModernBERT)",
            "expert_id": 0,
            "metadata": [0.2, 80.0, 0.1, 0.0, 0.1, 0.2, 0.1, 0.1, 0.0, 0.0, 0.0, 0.1, 0.1, 0.0, 0.1, 0.1],
            "payload": "La comunicazione semantica ottimizza l'efficienza delle reti 6G."
        },
        {
            "name": "STREAM AUDIO (Whisper-Tiny)",
            "expert_id": 1,
            "metadata": [5.0, 5060.0, 0.8, 0.5, 0.2, 1.0, 0.5, 0.4, 0.1, 0.2, 0.1, 0.5, 0.6, 0.3, 0.4, 0.2],
            "payload": np.random.uniform(-0.5, 0.5, 16000 * 2) # Stream audio simulato a 16kHz
        },
        {
            "name": "STREAM VIDEO (Google ViT)",
            "expert_id": 2,
            "metadata": [12.0, 1935.0, 0.5, -0.2, 0.1, 1.5, -0.4, 0.7, 0.2, -0.1, 0.0, 1.2, 0.4, -0.2, 0.3, -0.5],
            "payload": Image.open(FOTO_REAL_PATH).convert("RGB") if os.path.exists(FOTO_REAL_PATH) else None
        }
    ]
    
    # Eseguiamo il profiling del flusso continuo
    print("\n🔥 === INIZIO PROFILING HARDWARE DEL FLUSSO COMPLETO ===")
    for case in test_cases:
        if case["payload"] is None:
            continue
        # Lanciamo la simulazione di stream continuo (30 frame consecutivi)
        profiler.run_stream_test(moe_pipeline, decoder_semantico, case, num_frames=30)
        print("=" * 60)