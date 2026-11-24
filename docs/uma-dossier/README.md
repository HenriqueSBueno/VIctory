# Running a Unified-Memory Console Title on Windows: Hypotheses for VIctory

*A research dossier on why an Xbox Series (GDK, D3D12.X, RDNA2) title built for unified memory runs slowly and renders
black under VIctory on a discrete-GPU PC, and which approaches could fix it.*

- **Status:** working document, 24 November 2026. Hypotheses are labelled `H-<chapter>.<n>`; measurements come from our
  own runs on an RTX 3080 10 GB + i5-10400F, using our own legally purchased copy of the game. No game data
  (assets, code, disassembly, shader dumps or screenshots) is included, in line with VIctory's asset policy.
- **Method:** chapters were researched and drafted with AI assistance (Gemini via the Antigravity CLI and Claude Code)
  and reviewed by us. References were checked automatically: DOIs resolved through the Crossref API with title matching,
  URLs fetched; entries that could not be confirmed are marked `[unverified]`.
- **Discussion:** see the linked VIctory issue; please comment with results (confirmed / refuted / measured numbers).

## Contents

1. [Chapter 01 — Fundamentals: Unified vs. Discrete Memory Architectures and Why UMA Titles Collapse on Discrete-GPU Windows PCs](ch01.md)
2. [Chapter 02 — State of the Art: How Emulators and Translation Layers Handle Shared CPU/GPU Memory](ch02.md)
3. [Chapter 03 — Write Tracking and Software Coherence](ch03.md)
4. [Chapter 04 — Zero-Copy and Memory Mapping on Windows: From Shadow Copies to Address-Preserving GPU Backings](ch04.md)
5. [Chapter 05 — Textures, Tiling, Caching, Streaming, and VRAM Budgets](ch05.md)
6. [Chapter 06 — Shader Translation and SIMT Execution: From RDNA2 Wave64 to 32-Wide Host Warps](ch06.md)
7. [Chapter 07 — CPU–GPU Pipeline, Synchronization, Command Emulation, and Queue Virtualization](ch07.md)
8. [Chapter 08 — Queries, Timing, and Game-Side Adaptivity: Occlusion Culling, Timestamps, Dynamic Resolution, Frame Pacing, and Variable Rate Shading](ch08.md)
9. [Chapter 09 — Beyond Xbox and Discrete PCs: Unified-Memory APUs, CXL, ReBAR, Linux/Proton, Multi-Adapter, Cloud, and Platform Recommendations](ch09.md)
10. [Chapter 10 — Capstone Synthesis: Consolidated Hypothesis Matrix, Prioritized Experimental Plan, Phased Roadmap, and Master Reference Architecture](ch10.md)

*AI-assisted (Claude Code)*
