# AI Lab HSKA: Reinforcement Learning

Praktische Herangehensweise an die Lösung unterschiedlich komplexer Kontroll-Probleme mit Hilfe von modernen Reinforcement Learning (RL) Algorithmen. Von tabellarischen Methoden wie Q-Learning bis hin zur Funktionsapproximation durch Neuronale Netze soll versucht werden, Agenten in verschiedenen OpenAI Gym Umgebungen zu trainieren. Das Hauptziel am Ende ist es ein Atari Game mit Deep RL zu bewältigen.

## Vorbereitung

Docker Image bauen:
```bash
docker build -t ai-lab-rl .
```

Docker Container starten:
```bash
docker run -it --rm -v ~/ai-lab-hska-rl:/rl -p 8888:8888  ai-lab-rl
```
Im Browser `http://localhost:8888` aufrufen.
**Speichern klappt nur bei Tusted Notebooks!**

## Termine

| Datum | Inhalt |
|-|-|
| 31.05. | GridWorld mit Q-Learning |
| 07.06. | CartPole Gym mit Q-Learning |
| 21.06. | CartPole Gym mit DQN |
| 28.06. | Atari Gym mit DQN |
| 05.07. | Atari Gym mit DQN (Fortsetzung) |

## Hinweise

### Alle Ausgabezellen löschen

```bash
jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace Notebook.ipynb
```

## Weiterführende Lektüre

- "Standard"-Lektüre für den Einstieg in RL: [Reinforcement Learning: An Introduction (Richard S. Sutton and Andrew G. Barto)](http://incompleteideas.net/book/RLbook2018.pdf)
- Ausführlich und gut erklärter Einstieg in RL (Video-Lektionen) von David Silver (Google DeepMind): [UCL Course on RL (David Silver)](http://www0.cs.ucl.ac.uk/staff/D.Silver/web/Teaching.html)
- [Algorithms in Reinforcement Learning (Csaba Szepesvári)](https://sites.ualberta.ca/~szepesva/papers/RLAlgsInMDPs.pdf)
