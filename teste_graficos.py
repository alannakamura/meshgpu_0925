import pandas as pd
import matplotlib.pyplot as plt
import os

# Caminho para a pasta onde os arquivos de dados estão armazenados
pasta_dados = 'testes/gecco/testes041224'

# Obter a lista de arquivos na pasta
# arquivos = [f for f in os.listdir(pasta_dados) if f.endswith('.csv')]
arquivos = [f for f in os.listdir(pasta_dados)]
arquivos.sort()

# Configurações para criar uma grade de gráficos 2D
grade_linhas = 5
grade_colunas = 6

# Criar uma figura principal para a grade
fig, axs = plt.subplots(grade_linhas, grade_colunas, figsize=(18, 30))

# Adicionar rótulos à esquerda da grade
rotulos_esquerda = [f"DTLZ{i + 1}" for i in range(grade_linhas)]

# Loop para carregar e plotar os gráficos a partir dos arquivos
for i in range(grade_linhas):
    # if i not in [0, 2, 3]:
    for j in range(grade_colunas):
        if j not in [3]:
            index = i * grade_colunas + j  # Índice do arquivo de dados
            if index < len(arquivos):
                arquivo_atual = arquivos[index]
                caminho_arquivo = os.path.join(pasta_dados, arquivo_atual)

                # Ler os dados do arquivo CSV
                df = pd.read_csv(caminho_arquivo)

                # Verificar se as colunas necessárias estão presentes
                if {'x', 'y', 'z'}.issubset(df.columns):
                    x = df['x']
                    y = df['y']
                    z = df['z']

                    # Destacar pontos com valores altos em z
                    z_threshold = 0.9  # Limite para destacar pontos
                    highlight = z > z_threshold

                    # Selecionar o subplot
                    ax = axs[i, j]

                    # Adicionar pontos gerais (azuis) e destacados (vermelhos) ao gráfico
                    ax.scatter(x, y, c='blue', label='Pontos gerais', alpha=0.7)
                    ax.scatter(x[highlight], y[highlight], c='red', label='Maiores valores (z > 0.9)', s=50)

                    # Adicionar barras verticais ("sticks") para representar a altura de cada ponto
                    for k in range(len(x)):
                        ax.vlines(x[k], ymin=0, ymax=y[k], color='gray', linewidth=1)

                    # Configurar labels dos eixos com marcas em intervalos de 0.25
                    ax.set_xticks(np.arange(0, 1.25, 0.25))
                    ax.set_yticks(np.arange(0, 1.25, 0.25))

                    ax.set_xlabel('Eixo X', fontsize=10)
                    ax.set_ylabel('Eixo Y', fontsize=10)

                    # Adicionar título a cada gráfico
                    ax.set_title(f"Gráfico {index + 1}", fontsize=12)

                    # Adicionar rótulo da linha à esquerda da primeira coluna
                    if j == 0:
                        ax.text(-0.3, 0.5, rotulos_esquerda[i], transform=ax.transAxes, fontsize=14, va='center',
                                ha='center')
                else:
                    ax.set_title(f"Arquivo {arquivo_atual} não possui as colunas necessárias", fontsize=12)

# Ajustar espaçamento entre os subplots para evitar sobreposição
plt.tight_layout()

# Exibir a figura com a grade de gráficos 2D
plt.show()
