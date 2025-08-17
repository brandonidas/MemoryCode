#!/usr/bin/env python3
"""
Gemma 3 270M evaluation script for MemoryCode tiny dataset.
Optimized for sub-5K context dialogues using local Gemma model.
"""

import json
import os
import sqlite3
import signal
import sys
from datetime import datetime
from pathlib import Path
import requests
import fire
from tqdm import tqdm
import time

class GemmaEvaluator:
    def __init__(self, base_url="http://100.127.255.204:1234/v1", db_path="gemma_results.db"):
        self.base_url = base_url
        self.db_path = db_path
        self.interrupted = False
        self.model_name = "gemma-3-270m-it"  # Specific Gemma model
        self.init_database()
        
        # Setup signal handler for graceful interruption
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals gracefully"""
        print(f"\nReceived signal {signum}. Saving progress and stopping...")
        self.interrupted = True
    
    def init_database(self):
        """Initialize SQLite database for storing results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                dialogue_id INTEGER NOT NULL,
                session_id INTEGER NOT NULL,
                eval_type TEXT NOT NULL,
                prompt TEXT NOT NULL,
                response TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'pending',
                error_message TEXT,
                UNIQUE(run_id, dialogue_id, session_id, eval_type)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                dataset_dir TEXT NOT NULL,
                model_name TEXT,
                start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                end_time DATETIME,
                total_dialogues INTEGER,
                completed_dialogues INTEGER DEFAULT 0,
                status TEXT DEFAULT 'running'
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def call_gemma_api(self, prompt, preamble=None, max_retries=3):
        """Make API call to Gemma model with retry logic"""
        messages = []
        if preamble:
            messages.append({"role": "system", "content": preamble})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": -1,  # No limit, let model decide
            "stream": False
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    timeout=1200  # 20 minutes
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result["choices"][0]["message"]["content"]
                else:
                    print(f"API error (attempt {attempt + 1}): {response.status_code} - {response.text}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
            except Exception as e:
                print(f"Request error (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        
        raise Exception(f"Failed to get response after {max_retries} attempts")
    
    def extract_code(self, text):
        """Extract Python code from LM output"""
        import re
        # Try multiple code wrappers ```python and ```
        pattern = re.compile(r"\`\`\`python\n(.*?)\`\`\`", re.DOTALL)
        code_match = pattern.findall(text)
        if not code_match:
            pattern = re.compile(r"\`\`\`(.*?)\`\`\`", re.DOTALL)
            code_match = pattern.findall(text)
        extracted_text = code_match[0] if code_match else text
        return extracted_text
    
    def log_evaluation(self, run_id, dialogue_id, session_id, eval_type, prompt, response=None, error=None):
        """Log evaluation result to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        status = "completed" if response else "error"
        
        cursor.execute('''
            INSERT OR REPLACE INTO evaluations 
            (run_id, dialogue_id, session_id, eval_type, prompt, response, status, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (run_id, dialogue_id, session_id, eval_type, prompt, response, status, error))
        
        conn.commit()
        conn.close()
    
    def get_completed_evaluations(self, run_id):
        """Get set of completed evaluations for resume functionality"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT dialogue_id, session_id, eval_type 
            FROM evaluations 
            WHERE run_id = ? AND status = 'completed'
        ''', (run_id,))
        
        completed = set()
        for row in cursor.fetchall():
            completed.add((row[0], row[1], row[2]))
        
        conn.close()
        return completed
    
    def update_run_progress(self, run_id, completed_dialogues):
        """Update run progress in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE runs 
            SET completed_dialogues = ?, end_time = CASE WHEN ? = (SELECT total_dialogues FROM runs WHERE run_id = ?) THEN CURRENT_TIMESTAMP ELSE NULL END
            WHERE run_id = ?
        ''', (completed_dialogues, completed_dialogues, run_id, run_id))
        
        conn.commit()
        conn.close()
    
    def test_api_connection(self):
        """Test connection to Gemma API"""
        print("Testing connection to Gemma API...")
        try:
            test_response = self.call_gemma_api("Hello, can you respond with just 'OK'?")
            print(f"✅ API connection successful. Response: {test_response[:50]}...")
            return True
        except Exception as e:
            print(f"❌ API connection failed: {e}")
            print("Make sure the Gemma model is running on localhost:1234")
            return False
    
    def evaluate_dataset(self, dataset_dir="dataset_tiny", output_dir="outputs", run_id=None, resume=False):
        """Main evaluation function for tiny context dataset"""
        dataset_dir = Path(dataset_dir)
        output_dir = Path(output_dir)
        
        # Test API connection first
        if not self.test_api_connection():
            print("Exiting due to API connection failure")
            return
        
        # Generate run ID if not provided
        if not run_id:
            run_id = f"gemma_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Get dialogue files
        dialogue_files = sorted(list(dataset_dir.glob("dialogue_*.json")))
        total_dialogues = len(dialogue_files)
        
        if total_dialogues == 0:
            print(f"No dialogue files found in {dataset_dir}")
            return
        
        print(f"Starting Gemma evaluation with run_id: {run_id}")
        print(f"Found {total_dialogues} tiny context dialogues to evaluate")
        print(f"Model: {self.model_name}")
        print(f"Results will be saved to database: {self.db_path}")
        print(f"Press Ctrl+C to stop gracefully at any time\n")
        
        # Initialize run in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO runs (run_id, dataset_dir, model_name, total_dialogues, status)
            VALUES (?, ?, ?, ?, 'running')
        ''', (run_id, str(dataset_dir), self.model_name, total_dialogues))
        conn.commit()
        conn.close()
        
        # Get completed evaluations for resume
        completed = self.get_completed_evaluations(run_id) if resume else set()
        
        # Progress tracking
        completed_dialogues = 0
        total_evaluations = 0
        completed_evaluations = len(completed)
        
        # Count total evaluations needed
        for dialogue_file in dialogue_files:
            with open(dialogue_file, 'r') as f:
                dialogue = json.load(f)
            for session_id, session in enumerate(dialogue["sessions"]):
                if session.get("session_eval_query"):
                    total_evaluations += len(session["session_eval_query"])
                if session.get("history_eval_query"):
                    total_evaluations += len(session["history_eval_query"])
        
        # Create progress bar
        pbar = tqdm(
            total=total_evaluations,
            desc="Evaluating with Gemma",
            initial=completed_evaluations,
            unit="eval"
        )
        
        try:
            for dialogue_file in dialogue_files:
                if self.interrupted:
                    break
                
                dialogue_id = int(dialogue_file.stem.split('_')[1])
                
                # Load dialogue
                with open(dialogue_file, 'r') as f:
                    dialogue = json.load(f)
                
                dialogue_context = dialogue["context"]
                model_outputs = {"sessions": []}
                
                # Process each session
                for session_id, session in enumerate(dialogue["sessions"]):
                    if self.interrupted:
                        break
                    
                    session_output = {
                        "session_model_output": [],
                        "history_model_output": []
                    }
                    
                    # Session evaluations
                    session_eval_query = session.get("session_eval_query", [])
                    for eval_query in session_eval_query:
                        if self.interrupted:
                            break
                        
                        eval_key = (dialogue_id, session_id, f"session_{len(session_output['session_model_output'])}")
                        
                        if eval_key not in completed:
                            try:
                                # Create prompt optimized for Gemma
                                preamble = f"""You are {dialogue_context["mentee"]}, a software engineer at {dialogue_context["company"]}. Follow your mentor's coding guidelines exactly. Generate only Python code - no explanations or examples."""
                                
                                prompt = f"""Mentor {dialogue_context['mentor']} dialogue:
{session['text']}

Task: Write a {eval_query}. Follow all coding guidelines from the dialogue."""
                                
                                response = self.call_gemma_api(prompt, preamble)
                                session_output["session_model_output"].append(response)
                                
                                self.log_evaluation(run_id, dialogue_id, session_id, f"session_{len(session_output['session_model_output'])-1}", prompt, response)
                                
                            except Exception as e:
                                error_msg = str(e)
                                print(f"\nError in session evaluation: {error_msg}")
                                self.log_evaluation(run_id, dialogue_id, session_id, f"session_{len(session_output['session_model_output'])}", prompt, error=error_msg)
                                session_output["session_model_output"].append("")
                        else:
                            # Load from database
                            conn = sqlite3.connect(self.db_path)
                            cursor = conn.cursor()
                            cursor.execute('''
                                SELECT response FROM evaluations 
                                WHERE run_id = ? AND dialogue_id = ? AND session_id = ? AND eval_type = ?
                            ''', (run_id, dialogue_id, session_id, f"session_{len(session_output['session_model_output'])}"))
                            result = cursor.fetchone()
                            conn.close()
                            session_output["session_model_output"].append(result[0] if result else "")
                        
                        pbar.update(1)
                    
                    # History evaluations (only for last session)
                    history_eval_query = session.get("history_eval_query", [])
                    if session_id == len(dialogue["sessions"]) - 1:
                        # Build history sessions
                        history_sessions = ""
                        for i, sess in enumerate(dialogue["sessions"]):
                            history_sessions += f"\n\nSession {i}:\n{sess['text']}"
                        
                        for eval_query in history_eval_query:
                            if self.interrupted:
                                break
                            
                            eval_key = (dialogue_id, session_id, f"history_{len(session_output['history_model_output'])}")
                            
                            if eval_key not in completed:
                                try:
                                    preamble = f"""You are {dialogue_context["mentee"]}, a software engineer at {dialogue_context["company"]}. Follow your mentor's coding guidelines exactly. Generate only Python code - no explanations or examples."""
                                    
                                    prompt = f"""Full mentor {dialogue_context['mentor']} dialogue history:
{history_sessions}

Task: Write a {eval_query}. Follow all coding guidelines from the complete dialogue history."""
                                    
                                    response = self.call_gemma_api(prompt, preamble)
                                    session_output["history_model_output"].append(response)
                                    
                                    self.log_evaluation(run_id, dialogue_id, session_id, f"history_{len(session_output['history_model_output'])-1}", prompt, response)
                                    
                                except Exception as e:
                                    error_msg = str(e)
                                    print(f"\nError in history evaluation: {error_msg}")
                                    self.log_evaluation(run_id, dialogue_id, session_id, f"history_{len(session_output['history_model_output'])}", prompt, error=error_msg)
                                    session_output["history_model_output"].append("")
                            else:
                                # Load from database
                                conn = sqlite3.connect(self.db_path)
                                cursor = conn.cursor()
                                cursor.execute('''
                                    SELECT response FROM evaluations 
                                    WHERE run_id = ? AND dialogue_id = ? AND session_id = ? AND eval_type = ?
                                ''', (run_id, dialogue_id, session_id, f"history_{len(session_output['history_model_output'])}"))
                                result = cursor.fetchone()
                                conn.close()
                                session_output["history_model_output"].append(result[0] if result else "")
                            
                            pbar.update(1)
                    
                    model_outputs["sessions"].append(session_output)
                
                # Save model output to file
                output_dir.mkdir(exist_ok=True)
                gemma_output_dir = output_dir / "gemma"
                gemma_output_dir.mkdir(exist_ok=True)
                
                output_file = gemma_output_dir / f"output_{dialogue_id}.json"
                with open(output_file, 'w') as f:
                    json.dump(model_outputs, f, indent=2)
                
                completed_dialogues += 1
                self.update_run_progress(run_id, completed_dialogues)
                pbar.set_description(f"Gemma evaluation ({completed_dialogues}/{total_dialogues} dialogues)")
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            self.interrupted = True
        
        finally:
            pbar.close()
            
            # Update final status
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            status = "interrupted" if self.interrupted else "completed"
            cursor.execute('''
                UPDATE runs 
                SET status = ?, end_time = CURRENT_TIMESTAMP
                WHERE run_id = ?
            ''', (status, run_id))
            conn.commit()
            conn.close()
            
            print(f"\nGemma evaluation {status}!")
            print(f"Completed {completed_dialogues}/{total_dialogues} dialogues")
            print(f"Results saved in database: {self.db_path}")
            print(f"Output files saved in: {gemma_output_dir}")
            
            if self.interrupted:
                print(f"To resume, run: python gemma_evaluation.py evaluate_dataset --run_id {run_id} --resume")
    
    def show_results(self, run_id=None):
        """Show evaluation results from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if run_id:
            cursor.execute('''
                SELECT * FROM runs WHERE run_id = ?
            ''', (run_id,))
        else:
            cursor.execute('''
                SELECT * FROM runs ORDER BY start_time DESC LIMIT 10
            ''')
        
        runs = cursor.fetchall()
        
        print("Gemma evaluation runs:")
        print("-" * 80)
        for run in runs:
            print(f"Run ID: {run[0]}")
            print(f"Dataset: {run[1]}")
            print(f"Model: {run[2]}")
            print(f"Status: {run[7]}")
            print(f"Progress: {run[6]}/{run[5]} dialogues")
            print(f"Started: {run[3]}")
            if run[4]:
                print(f"Ended: {run[4]}")
            print("-" * 80)
        
        conn.close()

def main():
    """Main function for command line interface"""
    evaluator = GemmaEvaluator()
    fire.Fire(evaluator)

if __name__ == "__main__":
    main()
