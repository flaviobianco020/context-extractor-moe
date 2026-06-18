# src/utils/channel.py
import numpy as np

def apply_awgn_channel(semantic_vector, snr_db):
    """
    Simula il passaggio del vettore semantico attraverso un canale AWGN.
    Aggiunge rumore gaussiano bianco additivo in base all'SNR (in dB).
    """
    vector = np.array(semantic_vector, dtype=np.float32)
    
    # 1. Calcola la potenza del segnale semantico (Signal Power)
    signal_power = np.mean(vector ** 2)
    
    # 2. Converte l'SNR da decibel (dB) a scala lineare
    snr_linear = 10 ** (snr_db / 10.0)
    
    # 3. Calcola la potenza del rumore necessaria (Noise Power)
    noise_power = signal_power / snr_linear
    
    # 4. Genera il rumore gaussiano con media 0 e deviazione standard proporzionale alla potenza del rumore
    noise = np.random.normal(0, np.sqrt(noise_power), vector.shape)
    
    # 5. Il segnale ricevuto è la somma algebrica di Segnale originale + Rumore
    received_vector = vector + noise
    
    return received_vector.tolist()

def calculate_cosine_similarity(v1, v2):
    """
    Calcola la Cosine Similarity tra due vettori.
    Restituisce un valore tra -1 e 1 (dove 1 significa vettori semanticamente identici).
    """
    arr1 = np.array(v1)
    arr2 = np.array(v2)
    
    dot_product = np.dot(arr1, arr2)
    norm_v1 = np.linalg.norm(arr1)
    norm_v2 = np.linalg.norm(arr2)
    
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
        
    return float(dot_product / (norm_v1 * norm_v2))