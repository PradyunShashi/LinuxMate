==============================================================================
 LinuxMate - LLM Helper for Linux Commands & OS Lab Work   (Operating Systems Case Study)
 Team: 1. J. Kuladeep Adithya  24BCT0320     2. Pradyun Shashi  24BDS0052
==============================================================================

THE FILES
  linuxmate.py   Backend server (Flask + JSON knowledge base + Ollama + safety checker)
  knowledge.json 61 curated Q&A items across 8 categories (commands and OS concepts)
  history.json   Auto-created runtime query log & user feedback ratings
  .env           Settings (Ollama model, port)
  index.html     Frontend web application (Ask / Library / History / About)
  setup.sh       Helper script for WSL / Linux
  README.txt     This file

HOW IT WORKS
  Browser -> linuxmate.py -> knowledge.json (61 domain-specific Q&A pairs)
                          -> Ollama (llama3.2 or any local model)
  1. Your question -> matched against knowledge.json -> given to Ollama as worked examples.
  2. Answer format: Command / Meaning / Warning.
  3. Risky commands (rm -rf, mkfs, kill -9, chmod 777 ...) always receive a safety Warning.
  4. If Ollama is offline, the best matching knowledge base answer is returned directly.

------------------------------------------------------------------------------
 QUICK START
------------------------------------------------------------------------------
 Option A) Windows PowerShell:
        python linuxmate.py              (or: python linuxmate.py --check)
        Open http://localhost:5000 in your browser.

 Option B) WSL / Linux:
        bash setup.sh
        Open http://localhost:5000 in your browser.

------------------------------------------------------------------------------
 WHAT DO I NEED TO INSTALL?
------------------------------------------------------------------------------
 1. Flask:
        pip install flask
 2. Ollama (Optional for full AI generation; fallback answers work without it):
        Download installer from: https://ollama.com
        Run: ollama run llama3.2
=============================================================================
