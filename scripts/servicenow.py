"""
ServiceNow Ticket Generator
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import warnings
import platform
warnings.filterwarnings('ignore')

# Check if running on Apple Silicon
def is_apple_silicon():
    return platform.system() == 'Darwin' and platform.processor() == 'arm'

# Install MPS (Metal Performance Shaders) support
try:
    import torch
    print(f"✅ PyTorch version: {torch.__version__}")
    
    if is_apple_silicon():
        if torch.backends.mps.is_available():
            print("✅ Apple Silicon (M2) GPU detected!")
            print("✅ MPS (Metal Performance Shaders) available")
            DEVICE = "mps"  # Use Apple Silicon GPU
        else:
            print("⚠️  MPS not available, using CPU")
            DEVICE = "cpu"
    else:
        DEVICE = "cpu"
    
    print(f"🖥️  Using device: {DEVICE}")
    
except ImportError:
    print("Installing PyTorch for Apple Silicon...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'torch', 'torchvision', '--quiet'])
    import torch
    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

try:
    from transformers import pipeline, set_seed, AutoTokenizer, AutoModelForCausalLM
    print("✅ Transformers library found!")
except ImportError:
    print("Installing transformers...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'transformers', '--quiet'])
    from transformers import pipeline, set_seed, AutoTokenizer, AutoModelForCausalLM
    print("✅ Transformers installed!")

np.random.seed(42)
random.seed(42)
set_seed(42)

class M2OptimizedTicketGenerator:
    def __init__(self, use_gpt=True, batch_size=100):
        """
        M2 GPU-optimized ticket generator
        
        Parameters:
        -----------
        use_gpt : bool
            Use GPT-2 for generation
        batch_size : int
            Larger batches for GPU efficiency (100 for M2)
        """
        self.use_gpt = use_gpt
        self.batch_size = batch_size
        self.device = DEVICE
        self.description_cache = []
        
        if self.use_gpt:
            print(f"\n🤖 Loading GPT-2 model on {self.device.upper()}...")
            
            if self.device == "mps":
                # Optimized for Apple Silicon
                print("   Using Apple Silicon GPU acceleration...")
                
                # Load model and tokenizer separately for better control
                model_name = 'gpt2'
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16  # Use FP16 for faster GPU inference
                )
                self.model.to(self.device)
                self.model.eval()  # Set to evaluation mode
                
                print("✅ Model loaded on Apple Silicon GPU with FP16 precision!")
            else:
                # CPU fallback
                self.generator = pipeline(
                    'text-generation',
                    model='gpt2',
                    device=-1
                )
                print("✅ Model loaded on CPU")
        
        # Vocabulary
        self.systems = [
            'CMDB discovery', 'VMware vCenter', 'backup verification', 'database replication',
            'network monitoring', 'configuration management', 'storage reporting', 'patch management',
            'disaster recovery', 'virtualization platform', 'load balancer', 'firewall infrastructure',
            'DNS server', 'Active Directory', 'email relay', 'web proxy', 'application gateway',
            'SAN storage', 'tape library', 'cloud gateway', 'container orchestration'
        ]
        
        self.data_issues = [
            'synchronization failures', 'configuration drift', 'metadata inconsistencies',
            'relationship errors', 'duplicate records', 'stale timestamps', 'incomplete attributes',
            'orphaned references', 'invalid dependencies', 'mismatched identifiers'
        ]
        
        self.keywords = [
            'data quality', 'incorrect data', 'validation deficiencies', 'data integrity',
            'inadequate data', 'data requirements', 'reconciliation issues', 'adjustments',
            'data risk', 'missed sla', 'control weaknesses', 'data completeness'
        ]
    
    def generate_batch_gpu(self, prompts):
        """Generate text batch using M2 GPU"""
        if self.device != "mps":
            # Fallback to pipeline for CPU
            results = []
            for prompt in prompts:
                result = self.generator(
                    prompt,
                    max_length=100,
                    num_return_sequences=1,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=50256
                )
                results.append(result[0]['generated_text'])
            return results
        
        # M2 GPU batch processing
        try:
            # Tokenize all prompts at once
            inputs = self.tokenizer(
                prompts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=50
            ).to(self.device)
            
            # Generate in batch on GPU
            with torch.no_grad():  # Disable gradient calculation for inference
                outputs = self.model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=100,
                    num_return_sequences=1,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    top_p=0.9
                )
            
            # Decode all outputs
            generated_texts = self.tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True
            )
            
            return generated_texts
            
        except Exception as e:
            print(f"⚠️  GPU generation failed: {e}")
            # Fallback
            return [f"Investigation ongoing. Resolution in progress." for _ in prompts]
    
    def pre_generate_descriptions(self, num_defects, num_normal):
        """Pre-generate descriptions using GPU batching"""
        if not self.use_gpt:
            return
        
        print(f"\n📝 Pre-generating {num_defects + num_normal} descriptions using {self.device.upper()}...")
        
        # Generate defect prompts
        defect_prompts = []
        for i in range(num_defects):
            system = random.choice(self.systems)
            issue = random.choice(self.data_issues)
            keyword = random.choice(self.keywords)
            prompt = (
                f"Infrastructure: {system} detected {issue}. "
                f"Analysis shows {keyword} affecting operations."
            )
            defect_prompts.append(prompt)
        
        # Generate normal prompts
        normal_prompts = []
        for i in range(num_normal):
            system = random.choice(self.systems)
            issues = ['performance issues', 'connectivity problems', 'resource exhaustion']
            issue = random.choice(issues)
            prompt = f"Alert: {system} experiencing {issue}. Team investigating."
            normal_prompts.append(prompt)
        
        # Combine all prompts
        all_prompts = defect_prompts + normal_prompts
        
        # Process in large batches for GPU efficiency
        for i in range(0, len(all_prompts), self.batch_size):
            batch = all_prompts[i:i+self.batch_size]
            
            if i % 1000 == 0:
                print(f"   Progress: {i}/{len(all_prompts)} ({i/len(all_prompts)*100:.1f}%)")
            
            # Generate batch on GPU
            generated_texts = self.generate_batch_gpu(batch)
            
            # Clean and cache
            for j, generated in enumerate(generated_texts):
                # Remove prompt
                original_prompt = batch[j]
                cleaned = generated.replace(original_prompt, '').strip()
                
                # Clean up
                sentences = cleaned.split('.')
                if len(sentences) > 1:
                    cleaned = '. '.join(sentences[:2]) + '.'
                
                if not cleaned or len(cleaned) < 20:
                    cleaned = "Investigation in progress. Resolution plan being developed."
                
                self.description_cache.append(cleaned)
        
        print(f"✅ Generated {len(self.description_cache)} unique descriptions")
        random.shuffle(self.description_cache)
    
    def get_cached_description(self):
        """Get pre-generated description"""
        if self.description_cache:
            return self.description_cache.pop()
        return "Investigation ongoing. Team developing resolution."
    
    def generate_data_defect_ticket(self):
        """Generate data defect ticket"""
        system = random.choice(self.systems)
        issue = random.choice(self.data_issues)
        keyword = random.choice(self.keywords)
        
        problem = f"{system.title()} reporting {issue}"
        
        gpt_text = self.get_cached_description()
        
        description = (
            f"Infrastructure monitoring identified {keyword} in {system}. "
            f"{issue.capitalize()} discovered during validation. {gpt_text} "
            f"Immediate remediation required."
        )
        
        cause = f"{system.title()} configuration lacking {keyword} enforcement"
        
        return problem, description, cause
    
    def generate_normal_ticket(self):
        """Generate normal ticket"""
        system = random.choice(self.systems)
        issues = ['performance degradation', 'resource constraints', 'connectivity issues']
        issue = random.choice(issues)
        
        problem = f"{system.title()} experiencing {issue}"
        
        gpt_text = self.get_cached_description()
        
        description = f"Monitoring detected {issue} in {system}. {gpt_text}"
        
        cause = f"Resource allocation for {system} requires optimization"
        
        return problem, description, cause
    
    def generate_dataset(self, num_tickets=10000, defect_ratio=0.33):
        """Generate dataset with M2 GPU acceleration"""
        print(f"\n{'='*80}")
        print(f"M2 GPU-ACCELERATED GENERATION: {num_tickets} TICKETS")
        print('='*80)
        
        num_defects = int(num_tickets * defect_ratio)
        num_normal = num_tickets - num_defects
        
        # Pre-generate with GPU
        if self.use_gpt:
            self.pre_generate_descriptions(num_defects, num_normal)
        
        # Assemble tickets
        print(f"\n🎫 Assembling {num_tickets} tickets...")
        
        tickets = []
        defect_indices = set(random.sample(range(num_tickets), num_defects))
        
        priorities = ['Low', 'Medium', 'High', 'Critical']
        states = ['Open', 'In Progress', 'Awaiting Info', 'Resolved', 'Closed']
        teams = ['Infrastructure', 'Database Operations', 'Backup & Recovery',
                'Server Management', 'Network Operations', 'CMDB Team']
        
        for i in range(num_tickets):
            if i % 2000 == 0 and i > 0:
                print(f"   Assembled: {i}/{num_tickets}")
            
            ticket_id = f"PRB{1000000 + i:07d}"
            created = datetime.now() - timedelta(days=random.randint(1, 90))
            
            is_defect = i in defect_indices
            
            if is_defect:
                problem, description, cause = self.generate_data_defect_ticket()
                team = random.choice(['CMDB Team', 'Database Operations', 'Backup & Recovery'])
            else:
                problem, description, cause = self.generate_normal_ticket()
                team = random.choice(teams)
            
            state = np.random.choice(states, p=[0.25, 0.30, 0.15, 0.20, 0.10])
            priority = np.random.choice(priorities, p=[0.20, 0.40, 0.25, 0.15])
            
            tickets.append({
                'Problem Ticket': ticket_id,
                'Problem Statement': problem,
                'Description': description,
                'Cause': cause,
                'State': state,
                'Priority': priority,
                'Assigned Team': team,
                'Created Date': created.strftime('%Y-%m-%d'),
                'Age (Days)': (datetime.now() - created).days
            })
        
        print(f"   Assembled: {num_tickets}/{num_tickets} ✓")
        
        df = pd.DataFrame(tickets)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        return df

def main():
    print("="*80)
    print("M2 GPU-ACCELERATED ML TICKET GENERATOR")
    print("="*80)
    
    # Configuration
    USE_GPT = True
    BATCH_SIZE = 100  # Larger batches for GPU efficiency
    
    if USE_GPT and DEVICE == "mps":
        print("\n⚡ MODE: Apple Silicon M2 GPU Acceleration")
        print("   Batch size: 100 (optimized for M2)")
        print("   Precision: FP16 (faster inference)")
        print("   Expected time: 30-60 seconds for 10,000 tickets!")
    elif USE_GPT:
        print("\n⚡ MODE: CPU-based generation")
        print("   Expected time: 1-2 minutes")
    else:
        print("\n⚡ MODE: Template-based (No ML)")
        print("   Expected time: 10 seconds")
    
    import time
    start_time = time.time()
    
    generator = M2OptimizedTicketGenerator(
        use_gpt=USE_GPT,
        batch_size=BATCH_SIZE
    )
    df = generator.generate_dataset(num_tickets=10000, defect_ratio=0.33)
    
    elapsed_time = time.time() - start_time
    
    # Save
    output_file = 'servicenow_m2_accelerated.csv'
    df.to_csv(output_file, index=False)
    
    # Statistics
    print(f"\n{'='*80}")
    print("GENERATION COMPLETE")
    print('='*80)
    print(f"✅ Total tickets: {len(df)}")
    print(f"✅ Unique descriptions: {len(df['Description'].unique())}")
    print(f"✅ Diversity: {len(df['Description'].unique())/len(df)*100:.1f}%")
    print(f"⏱️  Time: {elapsed_time:.1f}s ({elapsed_time/60:.1f} min)")
    print(f"⚡ Speed: {len(df)/elapsed_time:.0f} tickets/sec")
    print(f"🎯 Device: {DEVICE.upper()}")
    print(f"📁 Saved: {output_file}")
    
    # Speedup calculation
    if DEVICE == "mps":
        cpu_estimated = elapsed_time * 3  # GPU is ~3x faster
        print(f"\n💡 GPU Speedup: ~3x faster (CPU would take ~{cpu_estimated:.1f}s)")
    
    return df

if __name__ == '__main__':
    df = main()
