with open('README.md', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace block 1
old1 = """```mermaid
flowchart LR
  User[Researcher / Operator] --> UI[Streamlit dashboard<br/>calibration_app.py]
  UI --> Core[quantum_optical_bus<br/>Python package]

  Core --> SF[Strawberry Fields<br/>Gaussian backend]
  Core -.->|optional| Meep["Meep (optional)"]

  Core --> UI
  Scripts[scripts/*] --> Assets[assets/web/* and assets/paper/* (PNG/GIF)]
```"""
new1 = """```mermaid
flowchart LR
  User["Researcher / Operator"] --> UI["Streamlit dashboard<br/>calibration_app.py"]
  UI --> Core["quantum_optical_bus<br/>Python package"]

  Core --> SF["Strawberry Fields<br/>Gaussian backend"]
  Core -.->|optional| Meep["Meep (optional)"]

  Core --> UI
  Scripts["scripts/*"] --> Assets["assets/web/* and assets/paper/* (PNG/GIF)"]
```"""

# Replace block 2
old2 = """```mermaid
flowchart TD
  UI[calibration_app.py] --> Map[interface.py<br/>P -> r]
  UI --> Units[units.py<br/>loss_dB -> T]

  Map --> SM[quantum.py<br/>run_single_mode]
  Units --> SM
  SM --> UI

  Map --> MM[multimode.py<br/>run_multimode]
  Units --> MM
  MM --> UI

  Map --> Topo[tdm_topology.py<br/>simulate_topology]
  Units --> Topo
  Topo --> UI

  UI --> Est[estimation.py<br/>fit_eta_and_loss]
  Est --> Ctrl[control.py<br/>latency + drift]
  Ctrl --> UI

  UI -.->|optional| HW["hardware.py<br/>Meep / analytic mock"]
  HW -.-> UI
```"""

new2 = """```mermaid
flowchart TD
  UI["calibration_app.py"] --> Map["interface.py<br/>P -> r"]
  UI --> Units["units.py<br/>loss_dB -> T"]

  Map --> SM["quantum.py<br/>run_single_mode"]
  Units --> SM
  SM --> UI

  Map --> MM["multimode.py<br/>run_multimode"]
  Units --> MM
  MM --> UI

  Map --> Topo["tdm_topology.py<br/>simulate_topology"]
  Units --> Topo
  Topo --> UI

  UI --> Est["estimation.py<br/>fit_eta_and_loss"]
  Est --> Ctrl["control.py<br/>latency + drift"]
  Ctrl --> UI

  UI -.->|optional| HW["hardware.py<br/>Meep / analytic mock"]
  HW -.-> UI
```"""

content = content.replace(old1, new1)
content = content.replace(old2, new2)

with open('README.md', 'w', encoding='utf-8') as f:
    f.write(content)
