# src/main.py
import torch
import os
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
    # 1. Allena il router sui metadati di rete simulati
    train_router()
    print("-" * 50)
    
    # 2. Inizializza la pipeline MoE con i pesi del router
    moe_pipeline = MoEPipeline(input_dim=16, router_weights_path="src/models/router_weights.pth")
    
    # 3. CONFIGURAZIONE DEI PERCORSI PER I FILE REALI
    FOTO_REAL_PATH = "data/raw/foto_test.jpg"
    
    print("\n📸 --- TEST SEMANTICO SU FOTO REALE ---")
    if not os.path.exists(FOTO_REAL_PATH):
        print(f"⚠️ ATTENZIONE: Inserisci una foto in {FOTO_REAL_PATH} per fare il test reale!")
    else:
        # Carica la vera immagine dal disco tramite Pillow
        real_image = Image.open(FOTO_REAL_PATH).convert("RGB")
        
        # Simuliamo i metadati di rete associati a un trasferimento d'immagine (instradato come VIDEO/IMAGE)
        video_metadata = [12.0, 1935.0, 0.5, -0.2, 0.1, 1.5, -0.4, 0.7, 0.2, -0.1, 0.0, 1.2, 0.4, -0.2, 0.3, -0.5]
        
        # Forward pass nell'architettura MoE
        output_foto = moe_pipeline.process_packet(video_metadata, real_image)
        
        # 4. PROTOCOLLO DI VALIDAZIONE DELL'OUTPUT SEMANTICO
        print("\n🔍 --- PROTOCOLLO DI VALIDAZIONE ---")
        semantic_data = output_foto["semantic_data"]
        
        # Controllo 1: Integrità della forma nello spazio latente
        is_length_valid = len(semantic_data) == 768
        # Controllo 2: Assenza di corruzione dei dati (NaN o valori nulli)
        is_data_corrupted = np.isnan(semantic_data).any()
        
        print(f"1. Controllo Lunghezza Spazio Latente (Atteso 768): {len(semantic_data)} -> {'✅ VALIDO' if is_length_valid else '❌ INVALIDO'}")
        print(f"2. Controllo Corruzione Dati (Assenza di NaN): {'✅ SUPERATO' if not is_data_corrupted else '❌ FALLITO'}")
        
        # Calcolo dell'efficienza di trasmissione sul canale
        original_size_mb = output_foto['original_bytes'] / 1e6
        transmitted_size_kb = output_foto['bytes'] / 1024
        compression_ratio = output_foto['original_bytes'] / max(1, output_foto['bytes'])
        
        print(f"3. Dimensione Immagine Grezza Origine: {original_size_mb:.2f} MB")
        print(f"4. Dimensione Vettore Semantico Estratto: {transmitted_size_kb:.2f} KB")
        print(f"🏆 Rapporto di Compressione Semantica: {compression_ratio:.2f}x")
        
        if is_length_valid and not is_data_corrupted:
            print("\n🟢 VERDETTO FINALE: L'OUTPUT È VALIDO PER LA TRASMISSIONE SEMANTICA!")
        else:
            print("\n🔴 VERDETTO FINALE: OUTPUT NON VALIDO. CONTROLLARE L'INPUT O IL BACKBONE.")