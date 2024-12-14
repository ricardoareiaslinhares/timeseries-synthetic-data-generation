# Modelos

Selecionei os modelos com melhores resultados no meus testes com a base de dados PAMAP2:

__1. CTGAN__
- Melhoria na atividade deitado (ppt. 41)
- _Input_: Dados com as features extraídas.
- _Output_: Dados com as features extraídas.

__2. Timegan__
- Melhoria na atividade deitado e correr (ppt. 16 e 18);
- _Input_: Raw, separados por atividade
- _Output_: Sequenced data dessa atividade

__3. RP__
- Melhoria na atividade deitado e correr (ppt. 47);
- _Input_: Dados com as features extraídas.
- _Output_: Dados com as features extraídas.

__4. RPT__
- Melhoria na atividade descer as escadas (ppt. 49);
- _Input_: Raw, separados por atividade
- _Output_: Raw, separados por atividade

__5. DoppleGan__
- Melhoria na atividade sentado (ppt. 56);
- _Input_: dados sequenciados de uma atividade específica
- _Output_: dados sequenciados dessa atividade

__6. Real N.V.P.__
- Melhoria na atividade subir e descer as escadas - o maior impacto - (ppt. 60);
- _Input_: Raw, separados por atividade
- _Output_: Raw, separados por atividade

__7. Gaussian Noise__
- Melhoria na atividade subir as escadas (ppt. 65);
- _Input_: Raw, separados por atividade
- _Output_: Raw, separados por atividade

__8. Timegan_tsgm__
- Melhoria na atividade subir as escadas (ppt. 53);
- _Input_: dados sequenciados de uma atividade específica
- _Output_: dados sequenciados dessa atividade
- Nota: Não cheguei a usar com mais de 100epochs mas eu experimentava com esta implementação antes da outra.

# Utilizar os Modelos

> __Nota: Como os dados têm um tamanho muito elevado tive que os remover deste repositório, no entanto deixo o link para o dataset que utilizei. Os dados originais e os formatos processados que usei nos modelos podem ser recriados seguindo o ``processing_data.ipynb``__


A pasta ``./models``contém pastas com os nomes dos modelos, todos têm um ficheiro ``using_model.ipynb`` que está pronto para correr os modelos utilizando um exemplo de dados (PAMAP2) na pasta ``./data``, cada modelo já chama os dados préprocessados exatamente como precisa deles.

Quando é necessário instalar packages específicos eu deixo o link no topo do respetivo ``using_model.ipynb``.

Nota: Os gans (dopplegan, timegan, ctgan) demoram muito tempo a treinar.

# Testar o dados sintéticos
O notebook ``synthetic_data_testing.ipynb`` explicita o processo de testar dados sintéticos, como exemplo utilizei dados gerados por RealNVP.
Treinei um modelo nos dados reais e testei também em dados reais; depois testei em sintéticos (i.e., é natural que o teste em sintéticos deixe a desejar, mas deve ser bom o suficiente para indicar que os dados sintéticos se assemelham aos reias). Depois, o teste mais importante é combinar os dados reais e sintéticos e testar nos reais, o desempenho deste modelo deverá ser superior ao do que foi treinado só como real.

É possivel fazer outros testes ``tsne_analysis`` recebe dados sequenciados e permite visualizar uma redução das distribuições. Tem tammbém testes de predictive score e descriminative score já implementados em ``synthetic_data_evaluation.py``, mas raramente os utilizei, considerei que os primeiros já davam a informação necessária para avaliar a qualidade dos dados sintéticos nesta aplicação.

References per model:
1. SDV vault
2. https://github.com/zzw-zwzhang/TimeGAN-pytorch
3. e 4. Data Augmentation of Wearable Sensor Data for Parkinson’s Disease Monitor
5. [Dopplegan](https://github.com/gretelai/gretel-synthetics)
6. https://openreview.net/pdf?id=PpshD0AXfA
7. e 8. https://tsgm.readthedocs.io/en/latest/index.html 