# src/utils/benchmark.py
import time
import torch
import psutil
import os
import numpy as np

class SemanticBenchmark:
    """
    Gestisce il profiling avanzato dell'architettura MoE.
    Misura velocità (latenza, FPS) e consumo risorse (RAM, CPU).
    """
    def __init__(self):
        self.process = psutil.Process(os.getpid())

    def get_resource_usage(self):
        """Restituisce l'utilizzo attuale di RAM (in MB) e CPU (%)."""
        ram_usage = self.process.memory_info().rss / (1024 * 1024) # MB
        cpu_usage = psutil.cpu_percent(interval=None)
        return ram_usage, cpu_usage

    def run_stream_test(self, pipeline, decoder, test_case, num_frames=30):
        """
        Simula un flusso continuo di dati (es. uno stream a 30 frame)
        e raccoglie le metriche hardware ed evolutive.
        """
        print(f"\n⏱️ Inizio Stress-Test Flusso Continuo: {test_case['name']} ({num_frames} pacchetti)...")
        
        latenze_tx = []
        latenze_channel = []
        latenze_rx = []
        ram_samples = []
        cpu_samples = []
        
        # Sincronizzazione iniziale per PyTorch (Warm-up)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        
        # Normalizzazione preventiva del payload se necessario
        from src.utils.channel import normalize_vector, apply_awgn_channel
        
        for i in range(num_frames):
            # Campionamento risorse pre-frame
            ram_start, cpu_start = self.get_resource_usage()
            
            # 1. TIMING TRASMETTITORE (Encoder Semantico MoE)
            t0 = time.perf_counter()
            # Forziamo l'esperto corretto con un mock contestuale
            from unittest.mock import patch
            with patch('torch.argmax', return_value=torch.tensor([test_case['expert_id']])):
                output_tx = pipeline.process_packet(test_case["metadata"], test_case["payload"])
            if torch.cuda.is_available(): torch.cuda.synchronize()
            t1 = time.perf_counter()
            
            latenze_tx.append((t1 - t0) * 1000) # Convertito in millisecondi
            
            # 2. TIMING CANALE (AWGN + Normalizzazione standard a 10dB)
            t_ch_start = time.perf_counter()
            v_tx_norm = normalize_vector(output_tx["semantic_data"])
            v_rx = apply_awgn_channel(v_tx_norm, snr_db=10)
            t_ch_end = time.perf_counter()
            latenze_channel.append((t_ch_end - t_ch_start) * 1000)
            
            # 3. TIMING RICEVITORE (Decoder Semantico)
            t2 = time.perf_counter()
            report_rx = decoder.decode_and_validate(v_rx, v_tx_norm)
            t3 = time.perf_counter()
            latenze_rx.append((t3 - t2) * 1000)
            
            # Campionamento risorse post-frame
            ram_end, cpu_end = self.get_resource_usage()
            ram_samples.append(ram_end)
            cpu_samples.append(cpu_end)
            
        # CALCOLO STATISTICHE FINALI
        avg_tx = np.mean(latenze_tx)
        avg_ch = np.mean(latenze_channel)
        avg_rx = np.mean(latenze_rx)
        total_latency_per_packet = avg_tx + avg_ch + avg_rx
        max_fps = 1000 / total_latency_per_packet
        
        print(f"📊 === REPORT PROFILING PER: {test_case['name']} ===")
        print(f"   ⏱️ Velocità ed Elaborazione (Latenza Media):")
        print(f"      - Trasmettitore (MoE Encoder): {avg_tx:.2f} ms")
        print(f"      - Canale Fisico (AWGN):        {avg_ch:.2f} ms")
        print(f"      - Ricevitore (Decoder):        {avg_rx:.2f} ms")
        print(f"      - Latenza Totale End-to-End:   {total_latency_per_packet:.2f} ms")
        print(f"      🚀 Throughput Massimo Stimato:   {max_fps:.2f} FPS / Pacchetti al secondo")
        print(f"   💻 Consumo Risorse Hardware (Medie su Flusso):")
        print(f"      - Occupazione RAM di picco:    {np.max(ram_samples):.2f} MB")
        print(f"      - Carico CPU Medio:            {np.mean(cpu_samples):.1f} %")
        
        # Validazione di utilizzabilità in tempo reale
        # Nello standard video, il tempo reale richiede una latenza inferiore a 33ms per frame (30 FPS)
        if total_latency_per_packet <= 33.3:
            print("   🟢 VERDETTO HARDWARE: IDONEO AL TEMPO REALE! Ottima efficienza computazionale.")
        else:
            print("   🟡 VERDETTO HARDWARE: ISOLATO COLLO DI BOTTIGLIA. Richiede accelerazione hardware (MPS/GPU).")