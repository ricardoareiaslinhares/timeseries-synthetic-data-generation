# Synthetic Models for timeseries data generation (specially HAR)
## Modelos

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

## Utilizar os Modelos

> __Nota: Como os dados têm um tamanho muito elevado tive que os remover deste repositório, no entanto deixo o link para o dataset que utilizei. Os dados originais e os formatos processados que usei nos modelos podem ser recriados seguindo o ``processing_data.ipynb``__


A pasta ``./models``contém pastas com os nomes dos modelos, todos têm um ficheiro ``using_model.ipynb`` que está pronto para correr os modelos utilizando um exemplo de dados (PAMAP2) na pasta ``./data``, cada modelo já chama os dados préprocessados exatamente como precisa deles.

Quando é necessário instalar packages específicos eu deixo o link no topo do respetivo ``using_model.ipynb``.

Nota: Os gans (dopplegan, timegan, ctgan) demoram muito tempo a treinar.

# Testar o dados sintéticos
O notebook ``synthetic_data_testing.ipynb`` explicita o processo de testar dados sintéticos, como exemplo utilizei dados gerados por RealNVP.
Treinei um modelo nos dados reais e testei também em dados reais; depois testei em sintéticos (i.e., é natural que o teste em sintéticos deixe a desejar, mas deve ser bom o suficiente para indicar que os dados sintéticos se assemelham aos reias). Depois, o teste mais importante é combinar os dados reais e sintéticos e testar nos reais, o desempenho deste modelo deverá ser superior ao do que foi treinado só como real.

É possivel fazer outros testes ``tsne_analysis`` recebe dados sequenciados e permite visualizar uma redução das distribuições. Tem tammbém testes de predictive score e descriminative score já implementados em ``synthetic_data_evaluation.py``, mas raramente os utilizei, considerei que os primeiros já davam a informação necessária para avaliar a qualidade dos dados sintéticos nesta aplicação.


## Models

I selected the models with the best results in my tests with the PAMAP2 database:

__1. CTGAN__
- Improvement in lying down activity (ppt. 41)
- _Input_: Data with extracted features.
- _Output_: Data with features extracted.

__2. Timegan__
- Improvement in the lying down and running activity (ppt. 16 and 18);
- _Input_: Raw, separated by activity
- _Output_: Sequenced data of this activity

__3. RP__
- Improvement in the lying down and running activity (ppt. 47);
- _Input_: Data with extracted features.
- _Output_: Data with extracted features.

__4. RPT__
- Improving the activity of going down the stairs (ppt. 49);
- _Input_: Raw, separated by activity
- _Output_: Raw, separated by activity

__5. DoppleGan__
- Improved sitting activity (ppt. 56);
- _Input_: sequenced data from a specific activity
- _Output_: sequenced data from that activity

__6. Real N.V.P.__
- Improving the activity of going up and down the stairs - the biggest impact - (ppt. 60);
- _Input_: Raw, separated by activity
- _Output_: Raw, separated by activity

__7. Gaussian Noise__
- Improved stair climbing activity (ppt. 65);
- _Input_: Raw, separated by activity
- _Output_: Raw, separated by activity

__8. Timegan_tsgm__
- Improvement in the activity of climbing the stairs (ppt. 53);
- _Input_: sequenced data from a specific activity
- _Output_: sequenced data from this activity
- Note: I didn't get to use it with more than 100epochs but I experimented with this implementation before the other one.

## Use the Models

> Note: As the data is very large, I had to remove it from this repository, but I'll leave the link to the dataset I used. The original data and the processed formats I used in the models can be recreated by following ``processing_data.ipynb``


The ``./models`` folder contains folders with the names of the models, all of which have a ``using_model.ipynb`` file that is ready to run the models using a data example (PAMAP2) in the ``./data`` folder, each model already calls the preprocessed data exactly as it needs it.

When specific packages need to be installed, I leave the link at the top of the respective ``using_model.ipynb``.

Note: The gans (dopplegan, timegan, ctgan) take a long time to train.

## Test the synthetic data
The notebook ``synthetic_data_testing.ipynb`` explains the process of testing synthetic data, as an example I used data generated by RealNVP.
I trained a model on real data and tested it on real data, then I tested it on synthetic data (i.e. it's natural that the test on synthetic data leaves something to be desired, but it should be good enough to indicate that the synthetic data is similar to the real data). Then, the most important test is to combine the real and synthetic data and test on the real data. The performance of this model should be superior to that of the one trained on the real data alone.

It is possible to run other ``tsne_analysis`` tests on sequenced data, which allows you to see a reduction in the distributions. There are also predictive score and descriminative score tests already implemented in ``synthetic_data_evaluation.py``, but I rarely used them as I felt that the former already provided the information needed to evaluate the quality of the synthetic data in this application.

## References per model:
1. SDV vault
2. https://github.com/zzw-zwzhang/TimeGAN-pytorch
3. & 4. Data Augmentation of Wearable Sensor Data for Parkinson’s Disease Monitor
5. [Dopplegan](https://github.com/gretelai/gretel-synthetics)
6. https://openreview.net/pdf?id=PpshD0AXfA
7. & 8. https://tsgm.readthedocs.io/en/latest/index.html 