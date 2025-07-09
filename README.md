# Projeto Elasticidade

## Objetivo

Este projeto tem como objetivo estudar a elasticidade de produtos mais vendidos no dataset e analisar a capacidade elasticas desse grupo de produtos. Assim podendo apoiar em decisões de ajustes de preço. 

## Dataset

O Dataset utilizado está disponível nesta publicação do Git.

## Etapas do Projeto

### 1. Importação e visualização dos dados
```bash
#1- Importando Dataset

# Diretorio onde do CSV  
diretorio_csv = 'caminho'

# Lendo o Data Frame
df = pd.read_csv(diretorio_csv)

#Mostrado os tipos de dados
df.info ()

# Estatisticas gerais do dataframe
df.describe()
```

### 2. Tratamentos de dados e criação de variáveis
```bash
#2- Tratamentos de dados e criacao de variaveis

# Contando valores vazios por coluna
df.isna().sum()

print(df.head(10))

# Filtrando as linhas que tem datas validas
df = df[pd.to_datetime(df['Date'], errors='coerce').notna()]


#Tratando coluna de Qty igual zero
df = df[df['Qty'] != 0]
```
Como na nossa base não existe a coluna de PVM, tivemos que criar para termos no modelo. 

```bash
#Criando coluna de PVM
df['PVM'] = df['Amount']/df['Qty'] 


print(df[['Date', 'SKU_Nome','Categoria_Nome' ,'Amount', 'Qty', 'PVM', 'Preco_Unitario']].head(5))
```

### 3. Análise exploratória de dados (EDA)
```bash
#3- Analise exploratoria de dados (EDA)

#3.1-  Regressão log-log para cada um dos Top 10 SKUs
```
Nesta etapa estamos agrupando e somando a Receita de cada SKU para termos os top 10 em Receita.

```bash
# Agrupar por SKU e somar Receita
top_skus = df.groupby('SKU_Nome')['Amount'].sum().sort_values(ascending=False).head(10)
```

```bash
# Plotando essa soma
plt.figure(figsize=(10,6))
sns.barplot(x=top_skus.values, y=top_skus.index, hue=top_skus.index, palette='viridis', dodge= False, legend= False)
plt.title('Top 10 SKUs por Receita (Amount)')
plt.xlabel('Receita Total')
plt.ylabel('SKU_Nome')
plt.show()
```
Criando um df só com os Top 10 SKUs em Receita.

```bash
# Filtrar somente os Top 10 SKUs
top_sku_list = top_skus.index.tolist()
df_top = df[df['SKU_Nome'].isin(top_sku_list)]
```

Aqui estou plotando um boxplot para mostrar a dispersão de Preço Médio de Venda desses Top SKUs. O objetivo é comparar como esse preço varia entre si, ajudando a entender a dispersão, e consistência desses preço no dataset.

```bash
# Plotar distribuicao de precos
plt.figure(figsize=(12,6))
sns.boxplot(x='SKU_Nome', y='PVM', data=df_top, palette='coolwarm')
plt.title('Price Architecture dos Top 10 SKUs')
plt.xlabel('SKU_Nome')
plt.ylabel('PVM')
plt.show()
```
Com essa arquitetura percebemos variações entre SKUs, como por exemplo, Produto 333 e Produto 37 com medianas altas, ao contrário do Produto 6. 

Importante é a amplitude interquatil (tamanho da caixa) que indicam variabilidade de Preços. Produtos como Produto 24 e Produto 37 têm amplitude maior, sugerindo preços menos consistentes. Veja que o Produto 333 tem preços mais altos, porém uma menor amplitude, indicando menos volatilidade.

Outro ponto de análise, são os limites superiores e inferiores. Produtos como Produto 24 e Produto 37 têm alta dispersãp, podendo ser estratégias de preço mais variadas, promoções, ou diferentes regiões/canais com preços diferentes.

Conclui-se que produtos com grande variabilidade (ex: Produto 24) podem precisar de estratégia de preço mais consistentes para posicinamento e percepção de valor. Em contrapartida, produtos mais consistentes (ex: Produto 33) por ser bem posicionados tem o risco da elasticidade, ou seja, demanda cai com a mudança no preço.

### 4.Aplicação dos Modelos

```bash
# Garantir que não haja zero em preço ou quantidade
df_top = df_top[(df_top['PVM'] > 0) & (df_top['Qty'] > 0)]
df_top['log_Qty'] = np.log(df_top['Qty'])
df_top['log_PVM'] = np.log(df_top['PVM'])

# Função para rodar o modelo de regressão log-log
def regressao_log(df_sub):
    modelo = smf.ols('log_Qty ~ log_PVM', data=df_sub).fit()
    return pd.Series({
        'Elasticity': modelo.params['log_PVM'],
        'R2': modelo.rsquared
    })
```


```bash
# Aplicar para cada SKU
resultados = df_top.groupby('SKU_Nome').apply(regressao_log).reset_index()
print(resultados.sort_values('Elasticity'))  # ordenando pelo mais elástico
```
Veja aqui a tabela do resultado esperado:

| SKU_Nome     | Elasticity   | R2         | 
|--------------|--------------|------------|
| Produto 24   | -3.656602    | 0.442051   | 
| Produto 35   |  -3.118857   | 0.039106   |
| Produto 37   |  -2.262640   | 0.203347   |
| Produto 46   |  -1.941754   | 0.042652   |
| Produto  6   |   -1.322428  | 0.013010   |
| Produto 41   |  -0.524937   | 0.002547   |
| Produto 17   |   -0.185000  | 0.000272   |
| Produto 333  |  0.086780    | 0.000139   |
| Produto 161  |   0.552789   | 0.010143   |
| Produto 109  |   1.236165   | 0.004689   |

Sabe-se que Elasticity mede a variação percentual em uma variável independente em resposta a uma variação percentual de uma outra váriavel. No nosso caso, como a quantidade de demanda influencia no preço. O índice em negativo signfica relação inversa, ou seja, conforme o preço sobre, a demanda cai.

E o número de R2, coeficiente de determinação, mede a eficiência da reta quanto aos dados manipulados. 

```bash
# Encontrar o SKU com maior elasticidade (mais negativo)
sku_mais_elastico = resultados.sort_values('Elasticity').iloc[0]['SKU_Nome']
df_sku_elastico = df_top[df_top['SKU_Nome'] == sku_mais_elastico]
```
 Aqui encontramos o SKU mais elástico, como já previsto, o Produto 24 por termos E= -3.65

```bash
# Plotar regressão
plt.figure(figsize=(8,6))
sns.regplot(x='log_PVM', y='log_Qty', data=df_sku_elastico, scatter_kws={'alpha':0.5})
plt.title(f'Regressão Log-Log para SKU {sku_mais_elastico} (Mais Elástico)')
plt.xlabel('log(PVM)')
plt.ylabel('log(Qty)')
plt.show()
```

Aqui plotamos o gráfico de Regressão do Produto 24, escolhido por ser nosso produto mais elástico do dataset. 


Veja que a linha tem coeficiente negativo, isso indica que um aumento no preço provoca um grande queda na quantidade de demanda.

Em números, sendo sua Elasticity = -3.65, aumentar o preço em 1% as vendas caem -%3,65. Em contrapartidade diminuir o preço em 1% as vendas aumento +3,65%.


### 5. Conclusão

Chegando ao fim desta análise de Elasticidade, verificamos, com este dataset utilizado conseguimos entender o comportamento de preço e demanda em uma amostra de produtos. 

Sabemos que seguindo essas etapas, conseguimos criar estratégias na prática em uma Gestão de Preços, como:

- Priorizar promoções agressivas para determinados mixes de produtos;

- Prever alta variabilidade de vendas conforme preço;

- Garantir níveis de estoque para atender o aumento de deamanda;

- Monitorar preços de concorrentes e posicionamento do seu produto no mercado.

Entre outras ações que poderão ser mapeadas e refinadas com uso de dados e modelos. 
