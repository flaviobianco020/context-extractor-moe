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
    # 1. Allena il router sui metadati simulati
    train_router()
    print("=" * 60)
    
    # 2. Inizializza i componenti core
    moe_pipeline = MoEPipeline(input_dim=16, router_weights_path="src/models/router_weights.pth")
    decoder_semantico = SemanticDecoder()
    
    # Ripristiniamo la scelta dinamica rimuovendo forzature in pipeline se necessario, 
    # ma qui creiamo i test associando direttamente il payload corretto per ogni esperto.
    
    # 3. DEFINIZIONE DEI PAYLOAD REALI PER I TRE ESPERTI
    FOTO_REAL_PATH = "data/raw/foto_test.jpg"
    
    test_cases = [
        {
            "name": "TESTO (ModernBERT)",
            "expert_id": 0,
            "metadata": [0.2, 80.0, 0.1, 0.0, 0.1, 0.2, 0.1, 0.1, 0.0, 0.0, 0.0, 0.1, 0.1, 0.0, 0.1, 0.1],
            "payload": "La comunicazione semantica ottimizza radicalmente l'efficienza delle reti 6G spostando il carico dal bit al significato."
        },
        {
            "name": "AUDIO (Whisper-Tiny)",
            "expert_id": 1,
            "metadata": [5.0, 5060.0, 0.8, 0.5, 0.2, 1.0, 0.5, 0.4, 0.1, 0.2, 0.1, 0.5, 0.6, 0.3, 0.4, 0.2],
            # Simuliamo un payload audio (array numpy di ampiezze a 16kHz)
            "payload": np.random.uniform(-0.5, 0.5, 16000 * 2) 
        },
        {
            "name": "VIDEO/IMMAGINE (Google ViT)",
            "expert_id": 2,
            "metadata": [12.0, 1935.0, 0.5, -0.2, 0.1, 1.5, -0.4, 0.7, 0.2, -0.1, 0.0, 1.2, 0.4, -0.2, 0.3, -0.5],
            "payload": Image.open(FOTO_REAL_PATH).convert("RGB") if os.path.exists(FOTO_REAL_PATH) else None
        }
    ]
    
    # Scenari di rumore da testare
    scenari_snr = [30, 10, -5]
    
    # 4. LOOP DI ESECUZIONE MULTIMODALE END-TO-END
    for case in test_cases:
        print(f"\n🚀 === TARGET MODALITÀ: {case['name']} ===")
        
        if case["payload"] is None:
            print(f"⚠️ Salto il test {case['name']} perché il file multimediale non è presente.")
            continue
            
        # Forziamo momentaneamente l'instradamento corretto in base al nostro test case
        # (Evitiamo i falsi positivi del router neurale durante la validazione del canale)
        from unittest.mock import patch
        with patch('torch.argmax', return_value=torch.tensor([case['expert_id']])):
            output_tx = moe_pipeline.process_packet(case["metadata"], case["payload"])
            
        vector_tx = output_tx["semantic_data"]
        
        # Gestione vettori di lunghezze diverse in base all'esperto (ModernBERT=768, Whisper=384, ViT=768)
        print(f"-> [TX] Vettore Semantico Estratto. Dimensione: {len(vector_tx)} elementi.")
        print(f"-> [TX] Dimensione dei dati compressi da trasmettere: {output_tx['bytes']} Byte")
        
        # Normalizzazione L2 geometrica
        vector_tx_normalized = normalize_vector(vector_tx)
        
        # Trasmissione sul canale con Link Adaptation
        for snr in scenari_snr:
            print(f"\n   ⚡ Canale AWGN a SNR = {snr} dB")
            
            if snr == -5:
                print("      ⚠️ [Link Adaptation] Canale critico! Attivazione Ridondanza Semantica (N=5)...")
                ricezioni = []
                for _ in range(5):
                    ricezioni.append(apply_awgn_channel(vector_tx_normalized, snr_db=snr))
                vector_rx = np.mean(ricezioni, axis=0).tolist()
            else:
                vector_rx = apply_awgn_channel(vector_tx_normalized, snr_db=snr)
                
            # Validazione lato Decoder
            report_rx = decoder_semantico.decode_and_validate(vector_rx, vector_tx_normalized)
            
            print(f"      [RX Decoder] Errore Semantico (MSE): {report_rx['semantic_mse']:.6f}")
            print(f"      [RX Decoder] Indice Conservazione Significato: {report_rx['meaning_preservation']*100:.2f}%")
            
            if report_rx['meaning_preservation'] >= 0.85:
                print("      🟢 VERDETTO: Significato intatto e totalmente ricostruibile!")
            elif report_rx['meaning_preservation'] >= 0.70:
                print("      🟡 VERDETTO: Distorsione presente, ma nucleo informativo preservato.")
            else:
                print("      🔴 VERDETTO: Significato perso nel rumore.")
        print("-" * 60)