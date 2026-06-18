# src/utils/channel.py
import numpy as np

def normalize_vector(vector):
    """
    Applica la normalizzazione L2 al vettore semantico 
    per portarlo a una norma unitaria (lunghezza = 1).
    """
    arr = np.array(vector, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm == 0:
        return arr.tolist()
    return (arr / norm).tolist()

def apply_awgn_channel(semantic_vector, snr_db):
    """
    Simula il passaggio del vettore semantico attraverso un canale AWGN.
    """
    vector = np.array(semantic_vector, dtype=np.float32)
    
    # 1. Calcola la potenza del segnale (che ora sarà stabile grazie alla normalizzazione)
    signal_power = np.mean(vector ** 2)
    
    # 2. Converte l'SNR da dB a scala lineare
    snr_linear = 10 ** (snr_db / 10.0)
    
    # 3. Calcola la potenza del rumore
    noise_power = signal_power / snr_linear
    
    # 4. Genera il rumore gaussiano
    noise = np.random.normal(0, np.sqrt(noise_power), vector.shape)
    
    # 5. Segnale ricevuto
    received_vector = vector + noise
    
    return received_vector.tolist()

def calculate_cosine_similarity(v1, v2):
    """
    Calcola la Cosine Similarity tra due vettori.
    """
    arr1 = np.array(v1)
    arr2 = np.array(v2)
    
    dot_product = np.dot(arr1, arr2)
    norm_v1 = np.linalg.norm(arr1)
    norm_v2 = np.linalg.norm(arr2)
    
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
        
    return float(dot_product / (norm_v1 * norm_v2))