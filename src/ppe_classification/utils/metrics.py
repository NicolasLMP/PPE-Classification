import pandas as pd
import matplotlib.pyplot as plt

species_list = [
    "Anthias", "Atherine", "Bar européen", "Bogue", "Carangue", "Daurade Royale",
    "Daurade rose", "Eperlan", "Girelle", "Gobie", "Grande raie pastenague", "Grande vive",
    "Grondin", "Maquereau", "Merou", "Mostelle", "Mulet cabot", "Muraine", "Orphie",
    "Poisson scorpion", "Rouget", "Sole commune"
]

csv_file = "fish_counts.csv"
df = pd.read_csv(csv_file, encoding='latin1')
df = df[df['ID'] != '---']

# Calculer le nombre de poissons pour chaque espèce
counts = []
for specie in species_list:
    counts.append(len(df[df['Espèce'] == specie]))

# Créer un DataFrame pour trier
plot_df = pd.DataFrame({'Espèce': species_list, 'Nombre': counts})
plot_df = plot_df.sort_values('Nombre', ascending=False)

# Création du graphique trié
fig, ax = plt.subplots(figsize=(16, 6))
bars = ax.bar(plot_df['Espèce'], plot_df['Nombre'], color='skyblue')

ax.set_ylabel('Nombre de détections')
ax.set_xlabel('Espèce')
ax.set_title('Distribution des espèces détectées')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()