# src/main.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.models.router import PacketRouter
from src.pipeline import MoEPipeline
from PIL import Image

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
    # Fase 1: Allena il router
    train_router()
    print("-" * 50)
    
    # Fase 2: Inizializza la pipeline MoE
    moe_pipeline = MoEPipeline(input_dim=16, router_weights_path="src/models/router_weights.pth")
    
    # =====================================================================
    # FASE 3: TEST DI INFERENZA SEMANTICA - CASO VIDEO
    # =====================================================================
    print("\n🔮 Test di inferenza Semantica su un pacchetto VIDEO simulato...")
    
    # Generiamo un frame video finto in alta risoluzione (1080p, Full HD RGB)
    # Un'immagine 1920x1080 a 3 canali colore (RGB)
    random_pixels = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    mock_video_frame = Image.fromarray(random_pixels)
    
    # Generiamo i metadati di rete affinché il Router capisca che è un VIDEO
    # (Vedi le regole definite nel training: dimensione grande ~10.0, porta 1935)
    sample_video_metadata = [11.4, 1935.0, 0.9, -0.1, 0.5, 2.2, -0.1, 0.8, 0.1, -0.5, 0.2, 1.9, 0.7, -0.1, 0.1, -0.6]
    
    output = moe_pipeline.process_packet(sample_video_metadata, mock_video_frame)
    
    # Analisi quantitativa dei consumi sul canale per l'ingegneria delle comunicazioni
    print("\n📊 --- REPORT DI COMUNICAZIONE SEMANTICA (VIDEO) ---")
    print(f"Tipo di Esperto Attivato: {output['expert'].upper()}")
    print(f"Dimensione del Frame Originale (Grezzo 1080p): {output['original_bytes']} Byte (~{output['original_bytes']/1e6:.2f} MB)")
    print(f"Dimensione delle Caratteristiche Semantiche ViT da Trasmettere: {output['bytes']} Byte (~{output['bytes']/1024:.2f} KB)")
    
    compression_ratio = output['original_bytes'] / max(1, output['bytes'])
    print(f"Rapporto di Compressione Semantica sul singolo Frame: {compression_ratio:.2f}x")
    print(f"Dati estratti pronti per il Canale: {str(output['semantic_data'])[:60]}...")