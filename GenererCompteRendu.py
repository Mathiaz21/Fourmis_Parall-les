import subprocess

result = subprocess.run("python3 ants.py", shell=True, capture_output=True, text=True)

contenu_compte_intro = """
# Compte Rendu OS202 - Projet Fourmis
## Mathias Gilbert

Ce compte rendu est généré intégralement et de manière automatique par le script python "GenererCompteRendu". **Attention** exécuter ce script écrase l'ancienne version du compte-rendu et en génère une nouvelle, dont les résultats dépendent des performances du pc exécutant le script.
"""

with open("CompteRenduMathiasGilbert.md", "w") as file:
    file.write(contenu_compte_intro)