# src/pipeline.py
import torch
from src.models.router.py import PacketRouter  # Se hai router.py dentro src/models
# Nota: se l'import sopra dà errore, usa: from src.models.router import PacketRouter
from src.models.experts import TextExpert, AudioExpert, VideoExpert

class MoEPipeline:
    """
    Orchestratore principale dell'architettura MoE.
    Riceve il pacchetto, interroga il Router e attiva solo l'esperto corretto.
    """
    def __init__(self, input_dim=16, router_weights_path=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Inizializzazione Pipeline MoE sul device: {self.device}")
        
        # 1. Inizializza il Router
        self.router = PacketRouter(input_dim=input_dim, num_classes=3).to(self.device)
        if router_weights_path:
            self.router.load_state_dict(torch.load(router_weights_path, map_location=self.device))
        self.router.eval()
        
        # 2. Inizializza gli Esperti (Caricamento Lazy/On-demand consigliato in seguito)
        self.text_expert = TextExpert(self.device)
        self.audio_expert = AudioExpert(self.device)
        self.video_expert = VideoExpert(self.device)

    def process_packet(self, packet_metadata, raw_payload):
        """
        Analizza i metadati del pacchetto ed estrae il contesto usando l'esperto designato.
        """
        with torch.no_grad():
            # Converte i metadati in un tensore per il Router
            tensor_meta = torch.tensor(packet_metadata, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Il router decide (0: Testo, 1: Audio, 2: Video)
            router_output = self.router(tensor_meta)
            selected_expert = torch.argmax(router_output, dim=1).item()
            
            # Instradamento dinamico
            if selected_expert == 0:
                print("🔀 Router: Assegnato all'esperto TESTO")
                context = self.text_expert.extract_context(raw_payload)
                return {"expert": "text", "context": context}
                
            elif selected_expert == 1:
                print("🔀 Router: Assegnato all'esperto AUDIO")
                context = self.audio_expert.extract_context(raw_payload)
                return {"expert": "audio", "context": context}
                
            elif selected_expert == 2:
                print("🔀 Router: Assegnato all'esperto VIDEO")
                context = self.video_expert.extract_context(raw_payload)
                return {"expert": "video", "context": context}