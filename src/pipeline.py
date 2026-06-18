# src/pipeline.py
import torch
from src.models.router import PacketRouter
from src.models.experts import TextExpert, AudioExpert, VideoExpert

class MoEPipeline:
    def __init__(self, input_dim=16, router_weights_path=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Inizializzazione Pipeline MoE sul device: {self.device}")
        
        self.router = PacketRouter(input_dim=input_dim, num_classes=3).to(self.device)
        if router_weights_path:
            self.router.load_state_dict(torch.load(router_weights_path, map_location=self.device))
        self.router.eval()
        
        self.text_expert = TextExpert(self.device)
        self.audio_expert = AudioExpert(self.device)
        self.video_expert = VideoExpert(self.device)

    def process_packet(self, packet_metadata, raw_payload):
        with torch.no_grad():
            tensor_meta = torch.tensor(packet_metadata, dtype=torch.float32).unsqueeze(0).to(self.device)
            router_output = self.router(tensor_meta)
            selected_expert = torch.argmax(router_output, dim=1).item()
            
            
            if selected_expert == 0:
                print("🔀 Router: Assegnato all'esperto TESTO")
                metrics = self.text_expert.extract_context(raw_payload)
                return {"expert": "text", **metrics}
                
            elif selected_expert == 1:
                print("🔀 Router: Assegnato all'esperto AUDIO")
                metrics = self.audio_expert.extract_context(raw_payload)
                return {"expert": "audio", **metrics}
                
            elif selected_expert == 2:
                print("🔀 Router: Assegnato all'esperto VIDEO")
                metrics = self.video_expert.extract_context(raw_payload)
                return {"expert": "video", **metrics}